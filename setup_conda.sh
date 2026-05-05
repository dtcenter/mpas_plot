# DO NOT RUN THIS SCRIPT AS A SHELL SCRIPT: IT MUST BE INVOKED USING THE source BUILTIN

# Check if conda location file exists
USE_SYSTEM_CONDA=false
if [ ! -f "conda_loc" ] && command -v conda &> /dev/null ; then
  CONDA_BASE=$(conda info --base)
  echo "Found existing conda installation at: ${CONDA_BASE}"
  read -p "Do you want to use your existing system conda? (y/n) " -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]] ; then
    USE_SYSTEM_CONDA=true
    echo "Using system conda installation..."
    # Initialize conda if not already initialized
    . "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
    echo "${CONDA_BASE}" > conda_loc
  else
    echo "Proceeding with local conda installation..."
  fi
else
  echo "No existing conda installation detected"
fi

if [ "$USE_SYSTEM_CONDA" = false ] ; then
  # Logic taken from UFS SRW Application (https://github.com/ufs-community/ufs-srweather-app)
  CONDA_BUILD_DIR="conda"
  echo "Building local conda install in ${CONDA_BUILD_DIR}/"
  os=$(uname)
  if [ ! -d "${CONDA_BUILD_DIR}" ] ; then
    test $os == Darwin && os=MacOSX
    hardware=$(uname -m)
    installer=Miniforge3-${os}-${hardware}.sh
    curl -L -O "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/${installer}"
    bash ./${installer} -bfp "${CONDA_BUILD_DIR}"
    rm -f ${installer}
  fi

  . ${CONDA_BUILD_DIR}/etc/profile.d/conda.sh
  # Put some additional packages in the base environment on MacOS systems
  if [ "${os}" == "MacOSX" ] ; then
    mamba install -y bash coreutils sed
  fi

  CONDA_BUILD_DIR="$(readlink -f "${CONDA_BUILD_DIR}")"
  echo "${CONDA_BUILD_DIR}" > conda_loc
  echo "Local conda build location: ${CONDA_BUILD_DIR}"

  if [[ ! "$PATH" =~ "$CONDA_BUILD_DIR" ]]; then
    export PATH=${CONDA_BUILD_DIR}/condabin:${CONDA_BUILD_DIR}/bin:${PATH}
  fi
  if [[ ! "$LD_LIBRARY_PATH" =~ "$CONDA_BUILD_DIR" ]]; then
    export LD_LIBRARY_PATH=${CONDA_BUILD_DIR}/lib:${LD_LIBRARY_PATH}
  fi
fi

conda activate
if ! conda env list | grep -q "^mpas_plot\s" ; then
  echo "Creating mpas_plot environment..."
  mamba env create -n mpas_plot --file environment.yml
else
  read -p "mpas_plot environment exists. Update it from environment.yml? (y/n) " -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]] ; then
    echo "Updating mpas_plot environment..."
    mamba env update -n mpas_plot --file environment.yml --prune
  fi
fi

conda activate mpas_plot
