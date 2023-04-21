#!/bin/bash

  if [ -f "multical.sh" ]; then
      chmod +x multical.sh
      ./multical.sh
  else
      echo $(ls .)
      echo "multical.sh not found"
      exit 1
  fi
