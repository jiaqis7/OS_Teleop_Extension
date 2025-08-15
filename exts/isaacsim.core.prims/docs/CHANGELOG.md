# Changelog

## [0.3.7] - 2025-01-26
### Changed
- Update test settings

## [0.3.6] - 2025-01-21
### Changed
- Update extension description and add extension specific test settings


## [0.3.5] - 2024-12-30
### Fixed
- Fixed cleanup for Prim class if it was not initialized correctly

## [0.3.4] - 2024-12-18
### Fixed
- handles_initialized returning the wrong value for a single articulation when articulation controller not initialized

## [0.3.3] - 2024-12-12
### Fixed
- Updated the implementation of articulation inverse dynamic functions to respect the type of articulation (floating base vs fixed based)

## [0.3.2] - 2024-11-19
### Fixed
- XformPrim._set_xform_properties to take in consideration scale:unitsResolve attribute if authored.

## [0.3.1] - 2024-11-08
### Changed
- Changed Articulation.invalidate_physics_callback and RigidPrim.invalidate_physics_callback to weakref callbacks.

## [0.3.0] - 2024-11-05
### Added
- Added joint_names argument to all the joint related methods for ease of specifying which joints to manipulate/ query.

## [0.2.0] - 2024-10-31
### Changed
- changed .initialize and .post_reset methods to be event triggered
- changed physX warmup and simulation context creation to be event trigerred

## [0.1.1] - 2024-10-24
### Changed
- Updated dependencies and imports after renaming


## [0.1.0] - 2024-10-18
### Added
- Initial release
