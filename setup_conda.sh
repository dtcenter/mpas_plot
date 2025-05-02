# DO NOT RUN THIS SCRIPT AS A SHELL SCRIPT: IT MUST BE INVOKED USING THE source BUILTIN

# Logic taken from UFS SRW Application (https://github.com/ufs-community/ufs-srweather-app)
CONDA_BUILD_DIR="conda"
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
conda activate
if ! conda env list | grep -q "^mpas_plot\s" ; then
  mamba env create -n mpas_plot --file environment.yml
fi

CONDA_BUILD_DIR="$(readlink -f "${CONDA_BUILD_DIR}")"

if [[ ! "$PATH" =~ "$CONDA_BUILD_DIR" ]]; then
  export PATH=${CONDA_BUILD_DIR}/condabin:${CONDA_BUILD_DIR}/bin:${PATH}
fi
if [[ ! "$LD_LIBRARY_PATH" =~ "$CONDA_BUILD_DIR" ]]; then
  export LD_LIBRARY_PATH=${CONDA_BUILD_DIR}/lib:${LD_LIBRARY_PATH}
fi

echo "To activate the mpas_plot environment, run this command:"
echo "  conda activate mpas_plot"
