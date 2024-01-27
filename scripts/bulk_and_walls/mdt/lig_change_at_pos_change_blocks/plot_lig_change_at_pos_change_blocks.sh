#!/bin/bash

settings="pr_nvt423_nh"
top_path="${HOME}/ownCloud/WWU_Münster/Promotion/Simulationen/results/lintf2_peo/walls"
cmp="Li-OE"

########################################################################
# Information and Usage Functions                                      #
########################################################################

information() {
    echo "Loop over all surface-simulation directories and run the"
    echo "Python script plot_lig_change_at_pos_change_block.py to plot"
    echo "the number of lithium-ion ligands that dissociate, associate"
    echo "or remain coordinated during the crossing of a free-energy"
    echo "barrier."
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
    echo "  -c    Compounds.  Default: '${cmp}'."
}

########################################################################
# Argument Parsing                                                     #
########################################################################

while getopts he:t:c: option; do
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
        c)
            cmp=${OPTARG}
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
py_script=$(readlink -e "${script_dir}/plot_lig_change_at_pos_change_block.py" || exit)
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

analysis="lig_change_at_pos_change_blocks"
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

        ana_dir="ana_${settings}_${system}/${tool}/${analysis}_${cmp}"
        if [[ ! -d ${ana_dir} ]]; then
            echo "WARNING: No such directory: '${ana_dir}'"
            cd ../../ || exit
            continue
        fi
        cd "${ana_dir}" || exit

        for barrier_dir in "${analysis}_${cmp}_"*; do
            if [[ ! -d ${barrier_dir} ]]; then
                echo "WARNING: No such directory: '${barrier_dir}'"
                continue
            fi
            cd "${barrier_dir}" || exit

            barrier="${barrier_dir#"${analysis}_${cmp}"_}"
            # shellcheck disable=SC2048,SC2086
            ${py_exe} "${py_script}" \
                --system "${system}" \
                --settings "${settings}" \
                --cmp "${cmp}" \
                --barrier "${barrier}" ||
                exit

            cd ../ || exit
        done

        cd ../../../../../ || exit
    done

    cd ../ || exit
    echo "============================================================="
done
cd "${cwd}" || exit

echo
echo "Done"
