; Inno Setup script for SpectroElectroChem Suite v3.0

#define MyAppName "SpectroElectroChem Suite"
#define MyAppVersion "3.0.0"
#define MyAppPublisher "Achim Habekost"
#define MyAppExeName "SpectroElectroChem_Suite.exe"

[Setup]
AppId={{8C7FD00E-5A90-4E6F-9A20-000000000030}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\SpectroElectroChem Suite
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\dist_installer
OutputBaseFilename=SpectroElectroChem_Suite_Setup_v3_0
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "..\dist\SpectroElectroChem_Suite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SpectroElectroChem Suite"; Filename: "{app}\SpectroElectroChem_Suite.exe"
Name: "{autodesktop}\SpectroElectroChem Suite"; Filename: "{app}\SpectroElectroChem_Suite.exe"

[Run]
Filename: "{app}\SpectroElectroChem_Suite.exe"; Description: "Launch SpectroElectroChem Suite"; Flags: nowait postinstall skipifsilent
