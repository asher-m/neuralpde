# Sea Ice Data
## Version 4
See [this](https://nsidc.org/data/g02202/versions/4) for more information on NOAA/NSIDC sea ice concentration data format 4.  In particular, the user manual is of significant aid.

### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V4/
cd data/V4/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/
```

## Version 5
See [this](https://nsidc.org/data/g02202/versions/5) for more information on NOAA/NSIDC sea ice concentration data format 5.  In particular, the user manual is of significant aid.

### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V5/
cd data/V5/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off https://noaadata.apps.nsidc.org/NOAA/G02202_V5/north/aggregate/
```
