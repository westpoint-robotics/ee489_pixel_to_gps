#!/bin/bash

set -o errexit

MY_PROMPT='$ '
MY_YESNO_PROMPT='(y/n)$ '

# version of the software
MAJOR_VERSION=1
MINOR_VERSION=7
RELEASE_TYPE=0
RELEASE_BUILD=6
INFORMATIONAL_VERSION=1.7.0.6
RELEASE_TYPE_TEXT=Release

echo "This is a script to assist with installation of the Spinnaker SDK.";
echo "Would you like to continue and install all the Spinnaker SDK packages?";
echo -n "$MY_YESNO_PROMPT"
read confirm

if [ $confirm = "n" ] || [ $confirm = "N" ] || [ $confirm = "no" ] || [ $confirm = "No" ]
then
    exit 0
    break
fi

echo

echo "Installing Spinnaker packages...";

sudo dpkg -i libspinvideoencoder-*.deb
sudo dpkg -i libspinnaker-*.deb
sudo dpkg -i libspinvideo-*.deb
sudo dpkg -i spinview-qt-*.deb
sudo dpkg -i spinupdate-*.deb
sudo dpkg -i spinnaker-*.deb

echo "Would you like to add a udev entry to allow access to USB hardware?";
echo "If this is not ran then your cameras may be only accessible by running Spinnaker as sudo.";
echo -n "$MY_YESNO_PROMPT"
read confirm

if [ $confirm = "n" ] || [ $confirm = "N" ] || [ $confirm = "no" ] || [ $confirm = "No" ]
then
    echo "Complete";
    exit 0
    break
fi

echo "Launching conf script";
sudo sh spin-conf

echo "Complete";

# Notify server of a linux installation
wget -T 10 -q --spider http://www.ptgrey.com/support/softwarereg.asp?text=ProductName+Linux+Spinnaker+$MAJOR_VERSION%2E$MINOR_VERSION+$RELEASE_TYPE_TEXT+$RELEASE_BUILD+%0D%0AProductVersion+$MAJOR_VERSION%2E$MINOR_VERSION%2E$RELEASE_TYPE%2E$RELEASE_BUILD%0D%0A

exit 0
