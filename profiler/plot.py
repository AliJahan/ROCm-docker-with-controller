import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from numpy import unravel_index

TICKS_LABEL_SIZE=19
TITLE_SIZE=20
AXIS_LABEL_SIZE=19
TITLE_PAD=-0.15
AXIS_LABEL_PAD=15
def get_data():
    array = np.loadtxt("profiling_data", delimiter=",", dtype=int)
    powers = sorted(list(set([i[2] for i in list(array)])))
    cus = sorted(list(set([i[1] for i in list(array)])))
    min_cus = cus[0]
    min_powers = powers[0]
    cus, powers = np.meshgrid(cus, powers)
    # print(powers.shape)
    avg_powers = np.zeros_like(powers, dtype=int)
    avg_tps = np.zeros_like(powers, dtype=int)
    for line in array:
        # print(line)
        j = (line[1]//min_cus)-1
        i = (line[2]//min_powers)-1
        avg_powers[i,j] = line[3]
        avg_tps[i,j] = line[4]
        # print(avg_powers)
        # input()
    avg_tps = avg_tps[-1,-1]/avg_tps
    return powers, cus, avg_powers, avg_tps
def plot_powers(fig, ax, powers, cus, avg_powers):
    # surf = ax.plot_wireframe(powers, cus, avg_powers, rstride=1, cstride=1)
    surf = ax.plot_surface(powers, cus, avg_powers, cmap=cm.RdYlGn_r,
                       linewidth=0, antialiased=False)#rstride=1, cstride=1)
    ax.set_proj_type('ortho')
    # ax.set_ylim(60, 2)
    ax.set_ylim(2, 60)
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    # Customize the z axis.
    ax.set_xlim(225, 25)
    ax.set_xlabel("Power Cap (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Number of CUs", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Power (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlim(0, 225)
    # ax.set_title("Power cap & number of CU\nvs\nPower")
    # size = fig.get_size_inches()*fig.dpi
    ax.set_title("a)\nPowerCap & NumCU vs Power",y=TITLE_PAD,fontsize=TITLE_SIZE)
def plot_diff(fig, ax, powers, cus, avg_powers):
    diff = np.abs(avg_powers - powers)
    print(dir(cm))
    surf = ax.plot_surface(powers, cus, diff, cmap=cm.RdYlGn_r,
                       linewidth=0, antialiased=False)#rstride=1, cstride=1)
    # surf = ax.plot_wireframe(powers, cus, avg_powers, rstride=1, cstride=2, edgecolor='r')
    ax.set_proj_type('ortho')
    # ax.set_ylim(60, 2)
    ax.set_ylim(2, 60)
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    # Customize the z axis.
    ax.set_xlim(225, 25)
    ax.set_xlabel("Power Cap (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Number of CUs", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Abs. Diff. (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlim(0, 225)
    max_ind = unravel_index(diff.argmax(), diff.shape)
    min_ind = unravel_index(diff.argmin(), diff.shape)
    # ax.set_zlim(diff[max_ind], diff[min_ind])
    ax.set_zlim(diff[min_ind], diff[max_ind])
    # ax.set_title("Power cap & number of CU\nvs\nPower")
    # size = fig.get_size_inches()*fig.dpi
    ax.set_title("b)\nAbs. Difference PowerCap & Power",y=TITLE_PAD,fontsize=TITLE_SIZE)
def plot_throughputs(fig, ax, powers, cus, avg_tps):

    # surf = ax.plot_wireframe(powers, cus, avg_tps, rstride=1, cstride=1)
    surf = ax.plot_surface(powers, cus, avg_tps, cmap=cm.RdYlGn,
                       linewidth=0, antialiased=False)#rstride=1, cstride=1)
    ax.set_proj_type('ortho')
    # ax.set_title("Power cap & number of CU\nvs\nNormalized TP Slowdown")
    # size = fig.get_size_inches()*fig.dpi
    ax.set_title("c)\nPowerCap & NumCU vs Norm. Throughput",y=TITLE_PAD,fontsize=TITLE_SIZE)
    # Customize the z axis.
    ax.set_xlim(225,25)
    # ax.set_xlim(25,225)
    # ax.set_ylim(60, 2)
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    ax.set_ylim(2, 60)
    ax.set_xlabel("Power Cap (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Number of CUs", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Norm. Throughput", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlim(0, avg_tps[-1,-1])

def plot_pe(fig, ax, powers, cus, pes):
    # surf = ax.plot_wireframe(powers, cus, pes, rstride=1, cstride=1)
    surf = ax.plot_surface(powers, cus, pes, cmap=cm.RdYlGn,
                       linewidth=0, antialiased=False)#rstride=1, cstride=1)
    ax.set_proj_type('ortho')

    # Customize the z axis.
    ax.set_xlim(225, 25)
    # ax.set_ylim(60, 2)
    ax.set_ylim(2, 60)
    ax.set_xlabel("Power Cap (W)", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Number of CUs", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    ax.set_zlabel("Norm. Power Efficiency", fontsize=AXIS_LABEL_SIZE,labelpad=AXIS_LABEL_PAD)
    max_ind = unravel_index(pes.argmax(), pes.shape)
    min_ind = unravel_index(pes.argmin(), pes.shape)
    # ax.set_title("Power cap & number of CU\nvs\nNormalized Power Efficiency\n(Normalized Power/Normalized TP Slowdown)")
    # size = fig.get_size_inches()*fig.dpi
    ax.tick_params(axis='x', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_LABEL_SIZE)
    ax.tick_params(axis='z', labelsize=TICKS_LABEL_SIZE)
    ax.set_title("d)\nPowerCap & NumCU vs Norm. Power Efficiency",y=TITLE_PAD,fontsize=TITLE_SIZE)
    ax.set_zlim(pes[min_ind], pes[max_ind])
    # ax.set_zlim(pes[max_ind], pes[min_ind])

def plot():
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), subplot_kw={"projection": "3d"})

    # Make data.
    powers, cus, avg_powers, avg_tps = get_data()
    plot_powers(fig, axs[0][0], powers, cus, avg_powers)
    plot_diff(fig, axs[0][1], powers, cus, avg_powers)
    plot_throughputs(fig, axs[1][0], powers, cus, avg_tps)
    avg_powers = avg_powers/avg_powers[-1,-1]
    plot_pe(fig, axs[1][1], powers, cus, (avg_tps.astype(float)/avg_powers.astype(float)))
    fig.tight_layout( rect=[-0.03, 0.1, .99, 1.0])
    plt.savefig("minimdock_cap_cu_profile.png")
    plt.savefig("minimdock_cap_cu_profile.pdf")



if __name__ == "__main__":
    plot()