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

class BEPowerProfilePlotter:
    def __init__(self, power_data, plot_save_dir: str, workload_name: str):
        self.power_data = power_data
        self.plot_save_dir = plot_save_dir
        self.workload_name = workload_name

    def plot(self):
        # prep plot layout
        fig, axs = plt.subplots(
            1,
            4,
            figsize=(31, 8),
            subplot_kw={"projection": "3d"}
        )

        # Read and pre-process data.
        powers, cus, avg_powers, avg_tps, min_cu, max_cu, min_power, max_power, cu_step, powers_step = self.process_power_profine()
        # Plot powers
        self.plot_powers(
            axs[0], powers, cus, avg_powers,
            min_cu, max_cu, min_power, max_power
        )
        # Plot precision
        self.plot_precision(
            axs[1], powers, cus, avg_powers,
            min_cu, cu_step, max_cu, min_power, powers_step, max_power
        )
        # Plot norm. throughput
        self.plot_throughputs(
            axs[2], powers, cus, avg_tps,
            min_cu, max_cu, min_power, max_power
        )
        # Plot norm. power efficiency
        avg_powers = avg_powers/max_power
        self.plot_pe(
            axs[3], powers, cus,
            (avg_tps.astype(float)/avg_powers.astype(float)),
            min_cu, cu_step, max_cu, min_power, powers_step, max_power
        )

        # Finalize for save
        fig.tight_layout(rect=[-0.03, 0.05, .99, .95])
        fig.legend(loc='upper center', ncol=4, fontsize=TITLE_SIZE+2)
        # Save in png and pdf in provided log_dir
        
        plt.savefig(f"{self.plot_save_dir}/{self.workload_name}.png")
        plt.savefig(f"{self.plot_save_dir}/{self.workload_name}.pdf")
        plt.close()

    # Read data from profiling_data file and pre-processes them
    def process_power_profine(self):
        # Read data into np.array (profiling_data file is expected to be in a log_dir)
        # array = np.loadtxt(f"{log_dir}/profiling_data", delimiter=",", dtype=float)
        
        # Get unique power and cu values
        powers = sorted(list(self.power_data.cap.unique()))
        cus = sorted(list(self.power_data.cu.unique()))
        
        # Calculate min max values
        min_cus = min(cus)
        cu_step = cus[1] - cus[0]
        max_cus = max(cus)
        min_powers = min(powers)
        powers_step = powers[1] - powers[0]
        max_powers = max(powers)

        # Create meshgrid for 3D plots
        cus, powers = np.meshgrid(cus, powers)
        avg_powers = np.zeros_like(powers, dtype=float)
        avg_tps = np.zeros_like(powers, dtype=float)
        
        # Pre-process read data
        for index, row in self.power_data.iterrows():
            if index == 0:
                continue
            j = int((row.cu-min_cus)//cu_step)
            i = int((row.cap-min_powers)//powers_step)
            avg_powers[i,j] = row.gpu_0_avg_pow
            # Throughput data are in form of sec/iteration, we reverse it 
            avg_tps[i,j] = row.avg_throughout

        # Other max, min stats
        max_ind = unravel_index(avg_tps.argmax(), avg_tps.shape)
        min_ind = unravel_index(avg_tps.argmin(), avg_tps.shape)
        max_tp = avg_tps[max_ind]
        min_tp = avg_tps[min_ind]
        # Normalized throughput
        avg_tps = avg_tps/max_tp

        return powers, cus, avg_powers, avg_tps, min_cus, max_cus, min_powers, max_powers, cu_step, powers_step
    # Plots power data
    def plot_powers(
            self,
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
        ax.set_zlim(min_power, max_power)

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
        self,
        ax,
        powers,
        cus,
        avg_powers,
        min_cu,
        cu_step,
        max_cu,
        min_power,
        power_step,
        max_power,
        precision_thres = 90
    ):
        # Compute the absolute difference
        diff = np.abs(avg_powers - powers)
        precision = 100.0 - (diff / max_power) * 100.0
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
        max_cu_high_prec = min_power
        for cap in range(len(avg_powers)):
            if precision[cap][-1] > precision_thres:
                max_cu_high_prec = min_power +(cap*power_step)
                break
        min_cu_high_prec = min_cu
        min_cap_high_prec = min_power

        found = False
        for cap in range(len(avg_powers)):
            for cu in range(len(avg_powers[cap])):
                if precision[cap][cu] > precision_thres:
                    min_cu_high_prec = min_cu+(cu*cu_step)
                    min_cap_high_prec = min_power +(cap*power_step)
                    found = True
                    break
            if found:
                break
        # Add green and red aread patched
        g_triangle = Poly3DCollection([
                (
                    (min_cap_high_prec, min_cu_high_prec, 0),#diff[max_ind]),
                    (max_cu_high_prec,60, 0),#diff[max_ind]),
                    (225,60, 0)#diff[max_ind])
                )
            ], 
            facecolors='g',
            edgecolors='g',
            linewidths=5,
            alpha=.3,
            label="High precision power control region"
        )
        r1_triangle = Poly3DCollection([
                (
                    (max_cu_high_prec-1,60, 0),
                    (min_power,max_cu, 0),
                    (min_cap_high_prec, min_cu_high_prec, 0)
                )
            ],
            facecolors='r', 
            edgecolors='r',
            linewidths=5,
            alpha=.3,
            zorder=0, 
            label="Low precision power control region"
        )
        r2_triangle = Poly3DCollection([
                (
                    (225 ,min_cu, 0),
                    (min_cap_high_prec-1, min_cu_high_prec-1, 0),
                    (224,59, 0)
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

        r1_triangle._facecolors2d=r1_triangle._facecolor3d
        r1_triangle._edgecolors2d=r1_triangle._edgecolor3d

        r2_triangle._facecolors2d=r2_triangle._facecolor3d
        r2_triangle._edgecolors2d=r2_triangle._edgecolor3d

        ax.add_collection(g_triangle)
        ax.add_collection(r1_triangle)
        ax.add_collection(r2_triangle)

        # Plot 3D precision plot
        
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
        self,
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
        self,
        ax,
        powers, 
        cus, 
        pes,
        min_cu,
        cu_step,
        min_power,
        power_step
    ):
        # Hand picked values for now TODO
        x = [175, 150, 125, 100, 75, 50]
        y = [54, 54, 54, 54, 54, 54]
        z = [ pes[(x[i]-min_power) // int(power_step), (y[i]-min_cu)//int(cu_step)] for i in range(len(x))]
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
        x = [50, 50, 50, 50, 50, 50]
        y = [54, 48, 42, 36, 30, 24]
        z = [ pes[(x[i]-min_power) // int(power_step), (y[i]-min_cu)//int(cu_step)] for i in range(len(x))]
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
        self,
        ax,
        powers,
        cus,
        pes,
        min_cu,
        cu_step,
        max_cu,
        min_power,
        power_step,
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
        self.draw_path(ax, powers, cus, pes, min_cu, cu_step, min_power, power_step)
        
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
