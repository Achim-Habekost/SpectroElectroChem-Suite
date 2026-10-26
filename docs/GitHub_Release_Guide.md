# GitHub release guide - v6.0.0

## Repository update

1. Replace/update the repository files with the contents of the v6.0.0 source package.
2. Commit the changes, for example: `Release v6.0.0 - ten-module suite`.
3. Verify that the default branch contains `__version__ = "6.0.0"`, the updated `CITATION.cff`, `README.md`, `CHANGELOG.md`, User Manual and build files.

## Create the release

Create tag **v6.0.0** from the updated default branch.

Recommended release title: **SpectroElectroChem Suite v6.0**

Use `RELEASE_NOTES_v6.0.0.md` as the release description.

The GitHub Actions workflow builds the Windows portable ZIP when the tag is pushed. The Windows installer can additionally be compiled from `installer/SpectroElectroChem_Suite_InnoSetup.iss` after the PyInstaller build.

## Recommended release assets

- `SpectroElectroChem_Suite_v6.0.0_Source.zip`
- `SpectroElectroChem_Suite_Windows_v6_0_0.zip` (GitHub Actions build)
- `SpectroElectroChem_Suite_Setup_v6_0_0.exe` (if built with Inno Setup)
- `User_Manual_v6_0.pdf`

## Zenodo

After publication of the GitHub release, archive the release through the connected Zenodo repository and verify the release metadata. Concept DOI: `10.5281/zenodo.21283231`.
