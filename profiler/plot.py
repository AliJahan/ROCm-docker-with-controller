import sys 
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import unravel_index
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Constants
OFFSET=5
TICKS_LABEL_SIZE=19+OFFSET
TITLE_SIZE=20+OFFSET
AXIS_LABEL_SIZE=19+OFFSET
TITLE_PAD=-0.08
AXIS_LABEL_PAD=15+OFFSET
POWER_ELEMENTS_TO_REMOVE=2


# Read data from profiling_data file and pre-processes them
def get_data(log_dir):
    # Read data into np.array (profiling_data file is expected to be in a log_dir)
    array = np.loadtxt(f"{log_dir}/profiling_data", delimiter=",", dtype=float)
    # Get unique power and cu values
    powers = sorted(list(set([i[2] for i in list(array)])))
    cus = sorted(list(set([i[1] for i in list(array)])))
    # Calculate min max values
    min_cus = min(cus)
    max_cus = max(cus)
    min_powers = min(powers)
    max_powers = max(powers)

    # Create meshgrid for 3D plots
    cus, powers = np.meshgrid(cus, powers)
    avg_powers = np.zeros_like(powers, dtype=float)
    avg_tps = np.zeros_like(powers, dtype=float)
    # Pre-process read data
    for line in array:
        j = int(line[1]//min_cus)-1
        i = int(line[2]//min_powers)-1
        avg_powers[i,j] = line[3]
        # Throughput data are in form of sec/iteration, we reverse it 
        avg_tps[i,j] = 1.0/float(line[4])

    # Other max, min stats
    max_ind = unravel_index(avg_tps.argmax(), avg_tps.shape)
    min_ind = unravel_index(avg_tps.argmin(), avg_tps.shape)
    max_tp = avg_tps[max_ind]
    min_tp = avg_tps[min_ind]
    # Normalized throughput
    avg_tps = avg_tps/max_tp

    return powers, cus, avg_powers, avg_tps, min_cus, max_cus, min_powers, max_powers, min_tp, max_tp
# Plots power data
def plot_powers(
        ax,
        powers,
        cus,
        avg_powers,
        min_cu,
        max_cu,
        min_power,
        max_power
    ):
    # Surface plot for powers
    surf = ax.plot_surface(
        powers, 
        cus, 
        avg_powers, 
        cmap=cm.RdYlGn_r,
        linewidth=0.5,
        antialiased=False
    )
    # Power capping only line
    cu60_power_keys = powers[:, 0]
    cu60_power_values = avg_powers[:, -1]
    ax.plot(
        cu60_power_keys,
        cu60_power_values,
        zs=60, # Place the line on y=60 (cu)
        zdir='y', 
        ls='dashed',
        lw=5,
        c='k',
        zorder=10,
        label="Power Cap Only"
    )
    ax.set_proj_type('ortho')

    # Axis limits
    ax.set_ylim(min_cu, max_cu)
    ax.set_xlim(max_power, min_power)
    ax.set_zlim(0, max_power)

    # Font sizes
    ## Tick labels
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    ## Axis labels
    ax.set_xlabel(
        "Power Cap (W)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_ylabel(
        "Num. GPU Cores",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_zlabel(
        "Consumed Power (W)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )

    # Title
    ax.set_title("a) Power model", y=TITLE_PAD, fontsize=TITLE_SIZE)
# Plots precision
def plot_precision(
    ax,
    powers,
    cus,
    avg_powers,
    min_cu,
    max_cu,
    min_power,
    max_power
):
    # Compute the absolute difference
    diff = np.abs(avg_powers - powers)
    # Prepare for calculating percentage
    max_ind = unravel_index(diff.argmax(), diff.shape)
    min_ind = unravel_index(diff.argmin(), diff.shape)
    # Power cap only (cu = 60) line
    cu60_power_keys = powers[:, 0]
    cu60_power_values = 100.0 - ( (diff[:, -1].astype(float)) / max_power ) * 100.0
    ax.plot(
        cu60_power_keys,
        cu60_power_values,
        zs=60,
        zdir='y',
        ls='dashed',
        lw=5,
        c='k',
        zorder=10
    )

    # Add green and red aread patched
    g_triangle = Poly3DCollection([
            (
                (25,0, 0),#diff[max_ind]),
                (25,60, 0),#diff[max_ind]),
                (225,60, 0)#diff[max_ind])
            )
        ], 
        facecolors='g',
        edgecolors='g',
        linewidths=5,
        alpha=.3,
        label="High precision power control region"
    )

    r_triangle = Poly3DCollection([
            (
                (225 ,0, 0),
                (25,0, 0),
                (225,60, 0)
            )
        ],
        facecolors='r', 
        edgecolors='r',
        linewidths=5,
        alpha=.3,
        zorder=0, 
        label="Low precision power control region"
    )

    g_triangle.set(linestyles="solid")
    g_triangle._facecolors2d=g_triangle._facecolor3d
    g_triangle._edgecolors2d=g_triangle._edgecolor3d

    r_triangle._facecolors2d=r_triangle._facecolor3d
    r_triangle._edgecolors2d=r_triangle._edgecolor3d

    ax.add_collection(g_triangle)
    ax.add_collection(r_triangle)

    # Plot 3D precision plot
    precision = 100.0 - (diff / max_power) * 100.0
    surf = ax.plot_surface(
        powers,
        cus,
        precision,
        cmap=cm.RdYlGn,
        linewidth=0,
        antialiased=False,
        zorder=10000
    )

    # surf = ax.plot_wireframe(powers, cus, avg_powers, rstride=1, cstride=2, edgecolor='r')
    
    ax.set_proj_type('ortho')
    
    # Axis limits
    ax.set_xlim(max_power, min_power)
    ax.set_ylim(min_cu, max_cu)
    ax.set_zlim(0, 100.0)

    # Font sizes
    ## Tick labels
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    
    ## Axis labels
    ax.set_xlabel(
        "Power Cap (W)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_ylabel(
        "Num. GPU Cores",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_zlabel(
        "Precision (%)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    # Title
    ax.set_title(
        "b) Power control vs. precision",
        y=TITLE_PAD,
        fontsize=TITLE_SIZE
    )
# Plot throughput
def plot_throughputs(
    ax,
    powers,
    cus,
    avg_tps,
    min_cu,
    max_cu,
    min_power,
    max_power
):
    # Power Capping plot
    cu60_power_keys = powers[:, 0]
    cu60_tp_values = avg_tps[:, -1]
    ax.plot(
        cu60_power_keys,
        cu60_tp_values,
        zs=60,
        zdir='y',
        ls='dashed',
        lw=5,
        c='k',
        zorder=10
    )
    # Plot 3D throughput plot
    # surf = ax.plot_wireframe(powers, cus, avg_tps, rstride=1, cstride=1)
    surf = ax.plot_surface(
        powers,
        cus,
        avg_tps,
        cmap=cm.RdYlGn,
        linewidth=0,
        antialiased=False
    )
    ax.set_proj_type('ortho')
    
    # Axis limits
    ax.set_xlim(max_power,min_power)
    ax.set_ylim(min_cu, max_cu)
    ax.set_zlim(0, 1.0)
    # Font sizes
    ## Tick labels
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    ## Axis labels
    ax.set_xlabel(
        "Power Cap (W)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_ylabel(
        "Num. GPU Cores",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    ax.set_zlabel(
        "Norm. Throughput",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=AXIS_LABEL_PAD
    )
    # Title
    ax.set_title(
        "c) Power control vs. throughput",
        y=TITLE_PAD,
        fontsize=TITLE_SIZE
    )

def draw_path(
    ax,
    powers, 
    cus, 
    pes,
    min_cu,
    min_power
):
    # Hand picked values for now TODO
    x = [175, 150, 125, 100, 75, 50]
    y = [56, 56, 56, 56, 56, 56]
    z = [ pes[x[i] // int(min_power), y[i]//int(min_cu)] for i in range(len(x))]
    # Power cap Line
    ax.plot(
        x,
        y,
        z,
        ls='dotted',
        lw=7,
        c='k',
        zorder=10
    )
    x = [50, 50, 50, 50, 50, 50, 50, 50]
    y = [56, 50, 44, 40, 34, 30, 24, 20]
    z = [ pes[x[i] // int(min_power), y[i] // int(min_cu)] for i in range(len(x))]
    # CU Line
    ax.plot(
        x,
        y,
        z,
        ls='dotted',
        lw=7, c='k',
        zorder=10,
        label="Optimum path for power shaping"
    )

# Plots power efficiency 
def plot_pe(
    ax,
    powers,
    cus,
    pes,
    min_cu,
    max_cu,
    min_power,
    max_power
):
    # Power Capping plot
    cu60_power_keys = powers[:, 0]
    cu60_pe_values = pes[:, -1]
    ax.plot(
        cu60_power_keys,
        cu60_pe_values,
        zs=60,
        zdir='y',
        ls='dashed',
        lw=5,
        c='k',
        zorder=10
    )
    # Plot 3D PE plot
    # surf = ax.plot_wireframe(powers, cus, pes, rstride=1, cstride=1)
    surf = ax.plot_surface(
        powers,
        cus,
        pes,
        cmap=cm.RdYlGn,
        linewidth=0,
        antialiased=False
    )

    ax.set_proj_type('ortho')
    # Add path to the PE plot
    draw_path(ax, powers, cus, pes, min_cu, min_power)
    
    # Axis limits
    ax.set_xlim(max_power, min_power)
    ax.set_ylim(min_cu, max_cu)
    max_ind = unravel_index(pes.argmax(), pes.shape)
    min_ind = unravel_index(pes.argmin(), pes.shape)
    ax.set_zlim(pes[min_ind], pes[max_ind])
    # Font sizes
    ## Tick labels
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    ## Axis labels
    ax.set_xlabel("Power Cap (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Num. GPU Cores", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Norm. Power Efficiency", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    # Title
    ax.set_title(
        "d) Power control vs. power efficiency",
        y=TITLE_PAD,
        fontsize=TITLE_SIZE
    )

def main():
    # log_dir needed as arg
    if len(sys.argv) < 2:
        print("Provide the directory with `profiling_data` in it\n", flush=True)
        sys.exit(0)
    # log dir which contain file exactly names "profiling_data"
    plot_dir = str(sys.argv[1])
    
    # prep plot layout
    fig, axs = plt.subplots(
        1,
        4,
        figsize=(31, 8),
        subplot_kw={"projection": "3d"}
    )

    # Read and pre-process data.
    powers, cus, avg_powers, avg_tps, min_cu, max_cu, min_power, max_power, min_tp, max_tp = get_data(plot_dir)
    # Plot powers
    plot_powers(
        axs[0], powers, cus, avg_powers,
        min_cu, max_cu, min_power, max_power
    )
    # Plot precision
    plot_precision(
        axs[1], powers, cus, avg_powers,
        min_cu, max_cu, min_power, max_power
    )
    # Plot norm. throughput
    plot_throughputs(
        axs[2], powers, cus, avg_tps,
        min_cu, max_cu, min_power, max_power
    )
    # Plot norm. power efficiency
    avg_powers = avg_powers/max_power
    plot_pe(
        axs[3], powers, cus,
        (avg_tps.astype(float)/avg_powers.astype(float)),
        min_cu, max_cu, min_power, max_power
    )

    # Finalize for save
    fig.tight_layout(rect=[-0.03, 0.05, .99, .95])
    fig.legend(loc='upper center', ncol=4, fontsize=TITLE_SIZE+2)
    # Save in png and pdf in provided log_dir
    file_name = plot_dir.split("_")[0]
    plt.savefig(f"{plot_dir}/{file_name}.png")
    plt.savefig(f"{plot_dir}/{file_name}.pdf")

if __name__ == "__main__":
    main()