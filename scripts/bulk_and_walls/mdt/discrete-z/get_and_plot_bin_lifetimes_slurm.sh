#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=q0heuer,hims,normal
#SBATCH --job-name="get_and_plot_bin_lifetimes"
#SBATCH --output="get_and_plot_bin_lifetimes_slurm-%j.out"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=a_thum01@uni-muenster.de
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
# The above options are only default values that can be overwritten by
# command-line arguments

# This script requires HPC Submit Scripts (hpcss)
# (https://github.com/andthum/hpc_submit_scripts).

# analysis="get_and_plot_bin_lifetimes"
thisfile=$(basename "${BASH_SOURCE[0]}")
echo "${thisfile}"
start_time=$(date --rfc-3339=seconds || exit)
echo "Start time = ${start_time}"

########################################################################
# Argument Parsing                                                     #
########################################################################

bash_dir="${HOME}/Promotion/hpc_submit_scripts/bash" # Directory containing bash scripts used by this script
py_lmod="${HOME}/Promotion/hpc_submit_scripts/lmod/palma/2020b/python3-8-6.sh" # File containing the modules to load Python
py_exe="${HOME}/Promotion/lintf2_ether_ana_postproc/.venv/bin/python3" # Name of the Python executable
leap_path="${HOME}/Promotion/lintf2_ether_ana_postproc" # Path to the installation of lintf2_ether_ana_postproc.

echo -e "\n"
echo "Parsed arguments:"
echo "bash_dir          = ${bash_dir}"
echo "py_lmod           = ${py_lmod}"
echo "py_exe            = ${py_exe}"
echo "leap_path         = ${leap_path}"

if [[ ! -d ${bash_dir} ]]; then
    echo
    echo "ERROR: No such directory: '${bash_dir}'"
    exit 1
fi

echo -e "\n"
bash "${bash_dir}/echo_slurm_output_environment_variables.sh"

########################################################################
# Load required executable(s)                                          #
########################################################################

# shellcheck source=/dev/null
source "${bash_dir}/load_python.sh" "${py_lmod}" "${py_exe}" || exit

########################################################################
# Start the Analysis                                                   #
########################################################################

top_path="/scratch/tmp/a_thum01/Promotion/lintf2_peo"

for settings in pr_nvt303_nh pr_nvt423_nh; do
    echo -e "\n"
    echo "settings = ${settings}"
    echo "================================================================="
    bash "${leap_path}/scripts/bulk_and_walls/mdt/discrete-z/get_and_plot_bin_lifetimes.sh" \
        -e "${settings}" \
        -t "${top_path}" \
        -c "Li" ||
        exit
    echo "================================================================="

    echo -e "\n"
    echo "settings = ${settings}"
    echo "--continuous"
    echo "================================================================="
    bash "${leap_path}/scripts/bulk_and_walls/mdt/discrete-z/get_and_plot_bin_lifetimes.sh" \
        -e "${settings}" \
        -t "${top_path}" \
        -c "Li" \
        -f "--continuous" ||
        exit
    echo "================================================================="
done

########################################################################
# Cleanup                                                              #
########################################################################

end_time=$(date --rfc-3339=seconds)
elapsed_time=$(bash \
    "${bash_dir}/date_time_diff.sh" \
    -s "${start_time}" \
    -e "${end_time}")
echo -e "\n"
echo "End time     = ${end_time}"
echo "Elapsed time = ${elapsed_time}"
echo "${thisfile} done"
