# Changelog

## [2.0.9] - 2025-01-26
### Changed
- Update test settings

## [2.0.8] - 2025-01-21
### Changed
- Update extension description and add extension specific test settings

### Fixed
- security fix to change os.umask(0) to os.umask(0o777)

## [2.0.7] - 2025-01-08
### Fixed
- updated physics reference link

## [2.0.6] - 2024-12-12
### Fixed
- Added sensor UI extensions as test dependencies

## [2.0.5] - 2024-12-09
### Changed
- ROS asset separator and undid some hacks

## [2.0.4] - 2024-12-06
### Changed
- Updated OmniGraph menu naming

## [2.0.3] - 2024-12-05
### Changed
- Updated Nova carter path

## [2.0.2] - 2024-12-03
### Changed
- No more Isaac Utils reference
- Glyphs for the Create Asset menu

## [2.0.1] - 2024-11-26
### Changed
- Replicator new menu layout restructure

## [2.0.0] - 2024-10-25
### Removed
- New menu layout

## [1.1.2] - 2024-10-24
### Changed
- Updated dependencies and imports after renaming

## [1.1.1] - 2024-10-15
### Changed
- FIxes typo in Jetbot menu option

## [1.1.0] - 2024-10-04
### Removed
- References to PhysX ultrasonic sensor

## [1.0.0] - 2024-09-30
### Changed
- extension renamed to isaacsim.gui.menu

## [0.7.3] - 2024-09-05
### Removed
- Grid Room from menu to match documentation

## [0.7.2] - 2024-09-03
### Added
- Missing robots to the menu
### Fixed
- some naming to match documentation

## [0.7.1] - 2024-08-26
### Added
- Missing environments to the menu
### Fixed
- Generation of April Tag

## [0.7.0] - 2024-08-23
### Added
- Several new robots to robot menu
### Changed
- Modifies robot menu to have multiple submenus based on robot type
- Aligns robot menu ordering with documentation

## [0.6.0] - 2024-08-21
### Changed
- Adds original Franka to robot asset menu
- Renames Franka -> Franka (alt. fingers)
- Moves Ant to Quadruped menu
- Adds Black Grid and Curved Grid menu options.

## [0.5.0] - 2024-07-10
### Removed
- Deprecated omni.isaac.dofbot and removed its usage.

## [0.4.1] - 2024-05-28
### Fixed
- Robotiq asset links

## [0.4.0] - 2024-05-24
### Changed
- Added leatherback
### Fixed
- broken USD paths

## [0.3.1] - 2024-05-11
### Changed
- Renamed Transporter to iw.hub

## [0.3.0] - 2024-05-07
### Added
- Added menu options for the new robots
- Added humanoid robot section

## [0.2.3] - 2024-04-22
### Fixed
- Sensor menu tests failing due to sensor extension now creates sensors on play, play timeline first before checking for values

## [0.2.2] - 2024-03-07
### Fixed
- Tests failing due to short delay when clicking, increased delay from 10 to 32 (default)

## [0.2.1] - 2024-02-20
### Fixed
- Extra dependencies from tests being incorrectly imported

## [0.2.0] - 2024-01-30
### Changed
- Changed sensor menu tests to verify sensor functionality through the sensor interfaces
- Added golden image test for the environment

## [0.1.0] - 2024-01-19
### Added
- Initial version of extension, moving menu code out of isaacsim.core.utils extension
