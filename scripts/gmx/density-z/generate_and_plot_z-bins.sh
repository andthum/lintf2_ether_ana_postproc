#!/bin/bash

settings="pr_nvt423_nh"
top_path="${HOME}/ownCloud/WWU_Münster/Promotion/Simulationen/results/lintf2_peo"
cmp="Li"

########################################################################
# Information and Usage Functions                                      #
########################################################################

information() {
    echo "Loop over all bulk and surface simulation directories and run"
    echo "the Python scripts to generate and/or plot z-bins."
}

usage() {
    echo
    echo "Usage:"
    echo
    echo "Required arguments (at least one of the following must be"
    echo "specified):"
    echo "  -g    Run the scripts to generate bins (i.e."
    echo "        generate_z-bins_bulk.py for bulk simulations and"
    echo "        generate_z-bins.py for surface simulations)."
    echo "  -p    Run plot_z-bins.py to polt the bins together with the"
    echo "        density and free-energy profile of the compound"
    echo "        specifed with -c."
    echo
    echo "Optional arguments:"
    echo "  -h    Show this help message and exit."
    echo "  -e    String describing the used simulation settings."
    echo "        Default: '${settings}'."
    echo "  -t    Top-level simulation path containing all bulk and"
    echo "        surface simulations.  Default: '${top_path}'."
    echo "  -c    The compound whose density and free-energy profile to"
    echo "        use for plotting.  For surface simulations, the"
    echo "        maxima of the free-energy profile of this compound"
    echo "        are used to generate the bins.  Default: '${cmp}'."
}

########################################################################
# Argument Parsing                                                     #
########################################################################

generate=false
plot=false
while getopts gphe:t:c: option; do
    case ${option} in
        # Required arguments.
        g)
            generate=true
            ;;
        p)
            plot=true
            ;;
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

if [[ ${generate} == false ]] && [[ ${plot} == false ]]; then
    echo "ERROR: At least one of -g or -p must be specified."
    exit 1
fi
if [[ ! -d ${top_path} ]]; then
    echo "ERROR: No such directory: '${top_path}'"
    exit 1
fi

########################################################################
# Main Part                                                            #
########################################################################

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
project_root=$(readlink -e "${script_dir}/../../.." || exit)
py_exe=".venv/bin/python3"
py_exe="${project_root}/${py_exe}"
py_script_generate_bulk=$(
    readlink -e "${script_dir}/generate_z-bins_bulk.py" || exit
)
py_script_generate_surface=$(
    readlink -e "${script_dir}/generate_z-bins.py" || exit
)
py_script_plot=$(readlink -e "${script_dir}/plot_z-bins.py" || exit)
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
if [[ ${generate} == true ]] && [[ ! -f ${py_script_generate_bulk} ]]; then
    echo "ERROR: No such file '${py_script_generate_bulk}'"
    exit 1
fi
if [[ ${generate} == true ]] && [[ ! -f ${py_script_generate_surface} ]]; then
    echo "ERROR: No such file '${py_script_generate_surface}'"
    exit 1
fi
if [[ ${plot} == true ]] && [[ ! -f ${py_script_plot} ]]; then
    echo "ERROR: No such file '${py_script_plot}'"
    exit 1
fi

analysis="density-z"
tool="gmx"

bulk_dir="bulk"
if [[ ! -d ${bulk_dir} ]]; then
    echo "ERROR: No such directory '${bulk_dir}'"
fi
echo
echo "================================================================="
echo "${bulk_dir}"
cd "${bulk_dir}" || exit
for system in lintf2_[gp]*[0-9]*_[0-9]*-[0-9]*_sc80; do
    echo
    echo "${system}"
    if [[ ! -d ${system} ]]; then
        echo "WARNING: No such directory: '${system}'"
        continue
    fi
    cd "${system}/" || exit

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

    if [[ ${generate} == true ]]; then
        ${py_exe} "${py_script_generate_bulk}" \
            --system "${system}" \
            --settings "${settings}" ||
            exit
    fi
    if [[ ${plot} == true ]]; then
        ${py_exe} "${py_script_plot}" \
            --system "${system}" \
            --settings "${settings}" \
            --cmp "${cmp}" ||
            exit
    fi

    cd ../../../../../ || exit
done
cd ../ || exit

surface_dir="walls"
if [[ ! -d ${surface_dir} ]]; then
    echo "ERROR: No such directory '${surface_dir}'"
fi
echo
echo "================================================================="
echo "${surface_dir}"
cd "${surface_dir}" || exit
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
        cd "${system}/" || exit

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

        if [[ ${generate} == true ]]; then
            ${py_exe} "${py_script_generate_surface}" \
                --system "${system}" \
                --settings "${settings}" \
                --cmp "${cmp}" ||
                exit
        fi
        if [[ ${plot} == true ]]; then
            ${py_exe} "${py_script_plot}" \
                --system "${system}" \
                --settings "${settings}" \
                --cmp "${cmp}" ||
                exit
        fi

        cd ../../../../../ || exit
    done

    cd ../ || exit
    echo "============================================================="
done
cd ../ || exit

echo
echo "Done"
