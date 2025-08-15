# Changelog

## [1.0.8] - 2025-01-26
### Changed
- Update test settings

## [1.0.7] - 2025-01-21
### Changed
- Update extension description and add extension specific test settings

## [1.0.6] - 2025-01-09
### Fixed
- Initial behavior not applied if interval is greater than 0

### Added
- Util functions for adding scripts and triggering and waiting for behavior events
- SDG scenario test with golden images

## [1.0.5] - 2024-12-12
### Fixed
- Cleaning up randomizers by calling reset when destroyed during play

## [1.0.4] - 2024-12-05
### Fixed
- Fixed volume stack behavior phyisics material assignment (not needed on parent prim)
- Avoid delta_time=0.0 randomizations through orchestrator.step() stage updates
- Exposed variables removed from fabric as well (avoid UI still showing them)
- Fixed reset simulation case without any performed simulation

## [1.0.3] - 2024-11-27
### Fixed
- Fixed error when removing all the scripts widget during runtime

## [1.0.2] - 2024-11-06
### Fixed
- Fixed stacking bounding box drop area calculation for scaled assets
- Fixed get world rotation util function to use Gf.Transform to get the rotation in case of transforms with scale/shear
- Check for previous colliders/rigid bodies for assets before simulation

### Changed
- Set exposed variables default values

## [1.0.1] - 2024-10-31
### Fixed
- Fixed 'remove_empty_scopes' to check if the prim is valid before searching for Scope or GenericPrim types

## [1.0.0] - 2024-09-29
### Added
- Added initial behavior script based randomizers

