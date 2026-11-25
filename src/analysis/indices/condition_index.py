import xarray as xr


class ConditionIndex:
    """
    Vegetation Condition Index and Rainfall Condition Index.

    The monthly climatological minimum and maximum are calculated
    independently for each spatial pixel using a reference period.

    The class calculates:
    - Raw monthly Condition Index.
    - Optional trailing rolling mean of the Condition Index.

    It does not perform temporal resampling.
    """

    name = "condition_index"

    def __init__(self, ds: xr.Dataset) -> None:
        """
        Initialize the ConditionIndex processor.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable from which the
            condition index will be calculated.
        """

        if not isinstance(ds, xr.Dataset):
            raise TypeError("ds must be an xarray Dataset.")

        if "time" not in ds.dims:
            raise ValueError(
                "Input dataset must contain a time dimension."
            )

        self.ds = ds
        self.index = None

    @staticmethod
    def rolling_mean(
        da: xr.DataArray,
        rolling_window: int = 3,
    ) -> xr.DataArray:

        if not isinstance(rolling_window, int) or isinstance(rolling_window, bool):
            raise TypeError("rolling_window must be an integer.")

        if rolling_window < 1:
            raise ValueError("rolling_window must be >= 1")

        if "time" not in da.dims:
            raise ValueError("DataArray must have a time dimension")

        return da.rolling(
            time=rolling_window,
            min_periods=rolling_window,
            center=False,
        ).mean()

    @staticmethod
    def condition_index(
        da: xr.DataArray,
        reference: xr.DataArray = None,
    ) -> xr.DataArray:
        """
        Calculate monthly pixel-wise condition index.

        Parameters
        ----------
        da : xr.DataArray
            Observations to normalize.

        reference : xr.DataArray, optional
            Data defining the climatological reference period.
            If None, the complete input series is used.

        Returns
        -------
        xr.DataArray
            Monthly pixel-wise condition index.
        """

        if "time" not in da.dims:
            raise ValueError(
                "Input DataArray must contain a time dimension."
            )

        if reference is None:
            reference = da

        if "time" not in reference.dims:
            raise ValueError(
                "Reference DataArray must contain a time dimension."
            )

        # Calculate monthly climatological minimum and maximum.
        monmin = reference.groupby("time.month").min(
            dim="time",
            skipna=True,
        )

        monmax = reference.groupby("time.month").max(
            dim="time",
            skipna=True,
        )

        # Select the corresponding monthly climatology
        # for every observation in the input series.
        sameshape_monmin = monmin.sel(
            month=da["time"].dt.month
        ).drop_vars("month")

        sameshape_monmax = monmax.sel(
            month=da["time"].dt.month
        ).drop_vars("month")

        # Calculate the monthly climatological range.
        denominator = sameshape_monmax - sameshape_monmin

        # Avoid division by zero.
        denominator = denominator.where(denominator != 0)

        # Calculate the condition index.
        result = (
            100
            * (da - sameshape_monmin)
            / denominator
        )

        result.name = f"{da.name or 'variable'}_{ConditionIndex.name}"

        result.attrs = {
            "long_name": "Monthly Condition Index",
            "units": "%",
            "climatology": "Calendar-month pixel-wise minimum and maximum",
        }

        return result

    def fit(
        self,
        variable: str,
        rolling_window: int = 1,
        reference_period: tuple = None,
    ) -> None:
        """
        Calculate the condition index and optional rolling mean.

        Parameters
        ----------
        variable : str
            Name of the input variable in the dataset.

        rolling_window : int
            Number of consecutive observations used for the
            trailing rolling mean. Default is 1.

        reference_period : tuple, optional
            Start and end dates defining the climatological
            reference period, e.g.
            ("2010-01-01", "2020-12-01").

        Notes
        -----
        The raw condition index is always retained.

        If rolling_window > 1, an additional rolling index is
        calculated and stored in self.index.
        """

        if not isinstance(rolling_window, int) or isinstance(
            rolling_window, bool
        ):
            raise TypeError("rolling_window must be an integer.")

        if rolling_window < 1:
            raise ValueError(
                "rolling_window must be greater than or equal to 1."
            )

        if variable not in self.ds.data_vars:
            raise ValueError(
                f"Variable '{variable}' not found. "
                f"Available variables: {list(self.ds.data_vars)}"
            )

        print(f"Fitting {variable} Condition Index")

        da = self.ds[variable]

        # Define the reference climatology.
        if reference_period is not None:

            if len(reference_period) != 2:
                raise ValueError(
                    "reference_period must contain a start and end date."
                )

            start, end = reference_period

            reference = da.sel(
                time=slice(start, end)
            )

            if reference.sizes["time"] == 0:
                raise ValueError(
                    "Reference period contains no observations."
                )

        else:
            reference = da

        # Calculate raw monthly condition index.
        condition_index = self.condition_index(
            da,
            reference=reference,
        )

        condition_index.name = f"{variable}_{self.name}"

        output = {
            condition_index.name: condition_index
        }

        # Apply a trailing rolling mean when requested.
        if rolling_window > 1:

            rolling_index = self.rolling_mean(
                condition_index,
                rolling_window,
            )

            rolling_name = (
                f"{variable}_{self.name}_{rolling_window}"
            )

            rolling_index = rolling_index.rename(rolling_name)

            rolling_index.attrs = condition_index.attrs.copy()
            rolling_index.attrs.update({
                "long_name": (
                    f"{rolling_window}-month rolling "
                    "Condition Index"
                ),
                "rolling_window": rolling_window,
                "rolling_alignment": "trailing",
            })

            output[rolling_name] = rolling_index

        self.index = xr.Dataset(output)

        print(
            f"Fitted {variable} Condition Index "
            f"and stored at `obj.index`"
        )