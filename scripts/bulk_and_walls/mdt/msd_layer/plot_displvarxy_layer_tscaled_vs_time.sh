#!/bin/bash

settings="pr_nvt423_nh"
top_path="${HOME}/ownCloud/WWU_Münster/Promotion/Simulationen/results/lintf2_peo/walls"
cmp="Li"
msd_component="xy"

########################################################################
# Information and Usage Functions                                      #
########################################################################

information() {
    echo "Loop over all surface-simulation directories and run the"
    echo "Python script plot_displvarxy_layer_tscaled_vs_time.py to"
    echo "scale and plot the displacement variance of the lithium ions"
    echo "in the first layer at the right (negative) electrode such"
    echo "that it overlaps with the corresponding displacement variance"
    echo "in a bulk layer."
}

usage() {
    echo
    echo "Usage:"
    echo
    echo "Optional arguments:"
    echo "  -h    Show this help message and exit."
    echo "  -e    String describing the used simulation settings."
    echo "        Default: '${settings}'."
    echo "  -t    Top-level simulation path containing all surface"
    echo "        simulations.  Default: '${top_path}'."
}

########################################################################
# Argument Parsing                                                     #
########################################################################

while getopts he:t: option; do
    case ${option} in
        # Optional arguments.
        h)
            information
            usage
            exit 0
            ;;
        e)
            settings=${OPTARG}
            ;;
        t)
            top_path=${OPTARG}
            ;;
        # Handling of invalid options or missing arguments.
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d ${top_path} ]]; then
    echo "ERROR: No such directory: '${top_path}'"
    exit 1
fi

########################################################################
# Main Part                                                            #
########################################################################

cwd=$(pwd) # Current working directory.

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
project_root=$(readlink -e "${script_dir}/../../../.." || exit)
py_exe=".venv/bin/python3"
py_exe="${project_root}/${py_exe}"
py_script=$(readlink -e "${script_dir}/plot_displvarxy_layer_tscaled_vs_time.py" || exit)
if [[ ! -d ${script_dir} ]]; then
    echo "ERROR: No such directory '${script_dir}'"
    exit 1
fi
if [[ ! -d ${project_root} ]]; then
    echo "ERROR: No such directory '${project_root}'"
    exit 1
fi
if [[ ! -x ${py_exe} ]]; then
    echo "ERROR: No such executable '${py_exe}'"
    exit 1
fi
if [[ ! -f ${py_script} ]]; then
    echo "ERROR: No such file '${py_script}'"
    exit 1
fi

analysis="msd_layer"
tool="mdt"

cd "${top_path}" || exit
for surfq in q[0-9]*; do
    echo
    echo "============================================================="
    echo "${surfq}"
    if [[ ! -d ${surfq} ]]; then
        echo "WARNING: No such directory: '${surfq}'"
        continue
    fi
    cd "${surfq}" || exit

    for system in lintf2_[gp]*[0-9]*_[0-9]*-[0-9]*_gra_"${surfq}"_sc80; do
        echo
        echo "${system}"
        if [[ ! -d ${system} ]]; then
            echo "WARNING: No such directory: '${system}'"
            continue
        fi
        cd "${system}" || exit

        sim_dir="${settings}_${system}"
        if [[ ${system} != *_gra_* ]]; then
            sim_dir="07_${sim_dir}"
        elif [[ ${system} == *_gra_q0_* ]]; then
            sim_dir="09_${sim_dir}"
        else
            sim_dir="01_${sim_dir}"
        fi
        if [[ ! -d ${sim_dir} ]]; then
            echo "WARNING: No such directory: '${sim_dir}'"
            cd ../ || exit
            continue
        fi
        cd "${sim_dir}" || exit

        ana_dir="ana_${settings}_${system}/${tool}/${analysis}"
        if [[ ! -d ${ana_dir} ]]; then
            echo "WARNING: No such directory: '${ana_dir}'"
            cd ../../ || exit
            continue
        fi
        cd "${ana_dir}" || exit

        ana_sub_dir="${analysis}_${cmp}"
        if [[ ! -d ${ana_sub_dir} ]]; then
            echo "WARNING: No such directory: '${ana_sub_dir}'"
            cd ../../../../../ || exit
            continue
        fi
        cd "${ana_sub_dir}" || exit

        ${py_exe} "${py_script}" \
            --system "${system}" \
            --settings "${settings}" \
            --msd-component "${msd_component}" ||
            exit

        cd ../../../../../../ || exit
    done

    cd ../ || exit
    echo "============================================================="
done
cd "${cwd}" || exit

echo
echo "Done"
