call wsl --install
call wsl sudo apt install dos2unix
call wsl dos2unix bin/activate.sh
call wsl dos2unix bin/deactivate.sh
call wsl dos2unix bin/install.sh
call wsl dos2unix bin/notebook.sh
call wsl dos2unix bin/reinstall.sh
call wsl dos2unix bin/uninstall.sh
call wsl source bin/install.sh