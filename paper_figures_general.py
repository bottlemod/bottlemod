from task import *
from func import *
from ppoly import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.transforms import Bbox
from mpl_toolkits.axisartist.axislines import AxesZero

# for camera ready paper submission only type 1 and truetype fonts are allowed
plt.rc('pdf', fonttype=42)

def Test1():
    print('Test1 Start')
    out_cpu = [PPoly([0, 10000], [[1]])] * 3
    in_cpu = [PPoly([0, 100], [[3]]), PPoly([0, 60, 100], [[10, 1.5]]), PPoly([0, 40, 100], [[0.8, 10]])]
    out_data = [Func([0, 100], [[1, 0]])] * 3
    in_data = [Func([0, 100], [[1, 0]]), Func([0, 50, 100], [[20], [100]]), Func([0, 63.245553203367586639977870888654], [[0.025, 0, 0]])]

    # sanity adjustments
    for oc in out_cpu:
        oc.x[-1] = out_data[0](out_data[0].x[-1])

    endresult, finalbottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()

    if finalbottlenecks != [2, -3, 1, -1, -2, 0] and finalbottlenecks != [2, -3, -3, 1, -1, -2, 0]:
        print('Wrong bottlenecks')

    expected_result1 = '''  0.00 -  16.00: 0.025*x^^(2) + 0.0*x^^(1) + 0.0*x^^(0)
 16.00 -  33.00: 0.0*x^^(2) + 0.8*x^^(1) + -6.4*x^^(0)
 33.00 -  50.00: 0.0*x^^(2) + 0.0*x^^(1) + 20.0*x^^(0)
 50.00 -  60.00: 0.0*x^^(2) + 3.0*x^^(1) + -130.0*x^^(0)
 60.00 -  80.00: 0.0*x^^(2) + 1.5*x^^(1) + -40.0*x^^(0)
 80.00 - 100.00: 0.0*x^^(2) + 1.0*x^^(1) + 0.0*x^^(0)
'''

    expected_result2 = ''' -0.00 -  16.00: 0.025*x^^(2) + 0.0*x^^(1) + 0.0*x^^(0)
 16.00 -  28.28: 0.0*x^^(2) + 0.8*x^^(1) + -6.4*x^^(0)
 28.28 -  33.00: 0.0*x^^(2) + 0.8*x^^(1) + -6.399999999999999*x^^(0)
 33.00 -  50.00: 0.0*x^^(2) + 0.0*x^^(1) + 20.0*x^^(0)
 50.00 -  60.00: 0.0*x^^(2) + 3.0*x^^(1) + -130.0*x^^(0)
 60.00 -  80.00: 0.0*x^^(2) + 1.5*x^^(1) + -40.0*x^^(0)
 80.00 - 100.00: 0.0*x^^(2) + 1.0*x^^(1) + 0.0*x^^(0)
'''

    if endresult.__str__() != expected_result1 and endresult.__str__() != expected_result2:
        print('Wrong result')

    print('Test1 End')

def Test1_Plot(save_pictures: bool = False):
    out_cpu = [PPoly([0, 20, 60, 10000], [[0.8, 4/3, 0.8/3]]), PPoly([0, 40, 90, 10000], [[0.4, 1, 0.3]]), PPoly([0, 30, 80, 10000], [[2, 0.8, 0.1]])]
    in_cpu = [PPoly([0, 100], [[4]]), PPoly([0, 60, 100], [[5, 1.5]]), PPoly([0, 40, 100], [[1.6, 6.5]])]
    out_data = [Func([0, 100], [[1, 0]])] * 3
    in_data = [Func([0, 100], [[1, 0]]), Func([0, 50, 100], [[20], [100]]), Func([0, 63.245553203367586639977870888654, 100], [[0.025, 0, 0], [0, 0, 100]])] #Func([0, 63.245553203367586639977870888654, 100], [[0.025, 0, 0], [0, 0, 100]])]

    # sanity adjustments
    for oc in out_cpu:
        oc.x[-1] = out_data[0](out_data[0].x[-1])

    endresult, finalbottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()

    ## plot some of the initial functions
    #plt.figure()
    #mpl.rc('lines', linestyle='solid')
    #for (i, f) in enumerate(in_cpu):
    #    xs = np.linspace(f.x[0], f.x[-1], 1000)
    #    color = 'C' + str(get_color_index(-1 - i, finalbottlenecks))
    #    plt.plot(xs, f(xs), color, label='CPU' + str(i), alpha=0.7)
    #plt.axis((0, 100, 0, 11))
    #plt.legend()
    #plt.xlabel('Time')
    #plt.ylabel('Assigned resource amount')
    #
    #plt.show()
    #exit(0)

    # potential output progress by storable resource inputs only
    fig = plt.figure()
    (result, bottlenecks) = TaskExecution.ppoly_min([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)])
    mpl.rc('lines', linestyle='solid')
    PlotPPoly(plt, result, bottlenecks, finalbottlenecks)
    mpl.rc('lines', linestyle='--')
    for (i, f) in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
        print(i, f)
        xs = np.linspace(f.x[0], f.x[-1], 1000)
        color = 'C' + str(get_color_index(i, finalbottlenecks))
        plt.plot(xs, f(xs), color, alpha=0.7)
    plt.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1])*1.02))
    plt.legend()
    plt.xlabel('time [time units]')
    plt.ylabel('potential progress [%]')
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 1, h * 0.4)
    plt.savefig('figures/dataprogress.pdf', bbox_inches='tight', pad_inches=0)

    ## final result only
    #plt.figure()
    #result, bottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()
    #print(result)
    #print(bottlenecks)
    #mpl.rc('lines', linestyle='solid')
    #PlotPPoly(plt, result, bottlenecks)
    #print('')
    #mpl.rc('lines', linestyle='--')
    #for (i, f) in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
    #    xs = np.linspace(f.x[0], f.x[-1], 1000)
    #    color = 'C' + str(get_color_index(i, bottlenecks))
    #    plt.plot(xs, f(xs), color, alpha=0.7)
    #plt.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1])*1.02))
    #plt.legend()
    #plt.ylabel('Final output progress')
    #plt.xlabel('Time')

#    # final result and unstorable resource usage subplots
#    _, (ax1, ax2) = plt.subplots(2, 1)
#
#    result, bottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()
#    print(result)
#    print(bottlenecks)
#    mpl.rc('lines', linestyle='solid')
#    PlotPPoly(ax1, result, bottlenecks)
#    print('')
#    mpl.rc('lines', linestyle='--')
#    for (i, f) in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
#        xs = np.linspace(f.x[0], f.x[-1], 1000)
#        color = 'C' + str(get_color_index(i, bottlenecks))
#        ax1.plot(xs, f(xs), color, alpha=0.7)
#    ax1.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1])*1.02))
#    ax1.legend()
#    ax1.set_ylabel('Final output progress')
#
##    # relative usage of unstorable resources in bottom subplot
##    #dresult = result.derivative()
##    #reciprocal_cpu_usage = [ic / oc(result) for (oc, ic) in zip(out_cpu, in_cpu)]
##    #mpl.rc('lines', linestyle='solid')
##    #xs = np.linspace(result.x[0], result.x[-1], 1000)
##    #for (i, f) in enumerate(reciprocal_cpu_usage):
##    #    color = 'C' + str(get_color_index(-1 - i, bottlenecks))
##    #    ax2.plot(xs, dresult(xs) / f(xs), color, label='CPU ' + str(i))
##    #ax2.axis((result.x[0], result.x[-1], 0, 1.02))
##    #ax2.legend()
##    #ax2.set_ylabel('Relative unstorable resource usage')
#
#    # absolute usage of unstorable resources in bottom subplot
#    dresult = result.derivative()
#    mpl.rc('lines', linestyle='solid')
#    xs = np.linspace(result.x[0], result.x[-1], 1000)
#    for (i, (inc, outc)) in enumerate(zip(in_cpu, out_cpu)):
#        color = 'C' + str(get_color_index(-1 - i, bottlenecks))
#        mpl.rc('lines', linestyle='--')
#        ax2.plot(xs, inc(xs), color)
#        mpl.rc('lines', linestyle='solid')
#        ax2.plot(xs, dresult(xs) * outc(result)(xs), color, label='Resource ' + str(i))
#    _, _, _, ymax = ax2.axis()
#    ax2.axis((result.x[0], result.x[-1], 0, ymax))
#    ax2.legend()
#    ax2.set_ylabel('Unstorable resource usage and potential')
#    plt.xlabel('Time')

#    # buffered yet unused input data
#    plt.figure()
#    mpl.rc('lines', linestyle='solid')
#    xs = np.linspace(result.x[0], result.x[-1], 1000)
#    for (i, (ind, outd)) in enumerate(zip(in_data, out_data)):
#        color = 'C' + str(get_color_index(i, bottlenecks))
#        ys = [ind(x) - outd.solve(result(x)) for x in xs]
#        plt.plot(xs, ys, color, label='Data ' + str(i))
#    _, _, _, ymax = plt.axis()
#    plt.axis((result.x[0], result.x[-1], 0, ymax))
#    plt.legend()
#    plt.xlabel('Time')
#    plt.ylabel('Unnecessary buffered input data')

#    # final result and buffered yet unused input data subplots
#    _, (ax1, ax2) = plt.subplots(2, 1)
#
#    result, bottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()
#    print(result)
#    print(bottlenecks)
#    mpl.rc('lines', linestyle='solid')
#    PlotPPoly(ax1, result, bottlenecks)
#    print('')
#    mpl.rc('lines', linestyle='--')
#    for (i, f) in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
#        xs = np.linspace(f.x[0], f.x[-1], 1000)
#        color = 'C' + str(get_color_index(i, bottlenecks))
#        ax1.plot(xs, f(xs), color, alpha=0.7)
#    ax1.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1])*1.02))
#    ax1.legend()
#    ax1.set_ylabel('Final output progress')
#
#    mpl.rc('lines', linestyle='solid')
#    xs = np.linspace(result.x[0], result.x[-1], 1000)
#    for (i, (ind, outd)) in enumerate(zip(in_data, out_data)):
#        color = 'C' + str(get_color_index(i, bottlenecks))
#        ys = [ind(x) - outd.solve(result(x)) for x in xs]
#        ax2.plot(xs, ys, color, label='Data ' + str(i))
#    _, _, _, ymax = plt.axis()
#    ax2.axis((result.x[0], result.x[-1], 0, ymax))
#    ax2.legend()
#    ax2.set_ylabel('Unnecessary buffered input data')
#    plt.xlabel('Time')

    # final result, unstorable resource usage and buffered yet unused input data subplots
    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig, (ax1, ax2) = plt.subplots(2, 1)

    result, bottlenecks = TaskExecution(Task(out_cpu, out_data), in_cpu, in_data).get_result()
    print(result)
    print(bottlenecks)
    mpl.rc('lines', linestyle='solid')
    PlotPPoly(ax1, result, bottlenecks)
    print('')
    mpl.rc('lines', linestyle='--')
    for (i, f) in enumerate([out_i(in_i) for (out_i, in_i) in zip(out_data, in_data)]):
        xs = np.linspace(f.x[0], f.x[-1], 1000)
        color = 'C' + str(get_color_index(i, bottlenecks))
        ax1.plot(xs, f(xs), color, alpha=0.7)
    ax1.axis((result.x[0], result.x[-1], result(result.x[0]), result(result.x[-1])*1.02))
    ax1.set_ylabel('progress [%]')
    ax1.tick_params(
        axis='both',
        which='both',
        bottom=True,
        top=False,
        labelbottom=False,
        left=False,
        right=True,
        labelleft=False,
        labelright=True
    )

    dresult = result.derivative()
    mpl.rc('lines', linestyle='solid')
    xs = np.linspace(result.x[0], result.x[-1], 1000)
    for (i, (inc, outc)) in enumerate(zip(in_cpu, out_cpu)):
        color = 'C' + str(get_color_index(-1 - i, bottlenecks))
        mpl.rc('lines', linestyle='--')
        ax2.plot(xs, inc(xs) * 100/6.5, color)
        mpl.rc('lines', linestyle='solid')
        ax2.plot(xs, [dresult(x) * outc(result)(x) * 100/6.5 + (i if x >= 33 and x < 50 else 0) for x in xs], color, label='resource' + str(i))
    _, _, _, ymax = ax2.axis()
    ax2.axis((result.x[0], result.x[-1], 0, ymax))
    ax2.set_ylabel('resources [%]')
    ax2.tick_params(
        axis='both',
        which='both',
        bottom=True,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
        labelright=True,
        right=True
    )

    #mpl.rc('lines', linestyle='solid')
    #xs = np.linspace(result.x[0], result.x[-1], 1000)[0:-1]
    #for (i, (ind, outd)) in enumerate(zip(in_data, out_data)):
    #    color = 'C' + str(get_color_index(i, bottlenecks))
    #    ys = [ind(x) - outd.solve(result(x)) for x in xs]
    #    if i == 1:
    #        ys = ys[:int(len(ys)//2)] + [y[0] + 1.35 for y in ys[int(len(ys)//2):]]
    #    ax3.plot(xs, np.array(ys) + 0.6, color, label='data' + str(i))
    #_, _, _, ymax = plt.axis()
    #ax3.axis((result.x[0], result.x[-1], 0, 102))
    #ax3.set_ylabel('buffered data [%]')
    #ax3.tick_params(
    #    axis='both',
    #    which='both',
    #    bottom=True,
    #    top=False,
    #    left=False,
    #    labelbottom=True,
    #    labelleft=False,
    #    labelright=True,
    #    right=True
    #)

    plt.xlabel('time [time units]')
    w, h = fig.get_size_inches()
    #fig.set_size_inches(w * 1.75, h * 1.2)
    fig.set_size_inches(w * 1.75, h * 0.8)
    handles, labels = ax1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    ax1.legend(handles, labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=6)
    plt.savefig('figures/finalprogressandmore.pdf', bbox_inches='tight', pad_inches=0.1)

    #plt.show()

def MorePaperFigures():
    dataOutput1 = PPoly([0, 100], [[1], [0]])
    dataOutput2 = PPoly([0, 97, 100], [[0, 100]])
    resourceOutput1 = PPoly([0, 100], [[0.1], [0]])
    resourceOutput2 = PPoly([0, 100], [[10]])
    xs = np.linspace(0, 100, 1000)

    fig = plt.figure()

    ax = fig.add_subplot(121, axes_class=AxesZero)
    ax.plot(xs, dataOutput1(xs), "C9", label='stream')
    ax.plot(xs, dataOutput2(xs), "C8", label='burst')

    for direction in ["left", "bottom"]: #["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")
        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)
        ax.axis[direction].toggle(all=False, label=True)

    for direction in ["right", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    ax.axis((0, 100, -0.75, 102))
    ax.set_xlabel('dependency\'s input data i')
    ax.set_ylabel('process\'s progress')
    ax.legend()
    #ax.text(-0.1, 1.15, '(a)', transform=ax.transAxes, size=12)
    ax.text(-0.1, 1.15, '(a) input data (storable)', transform=ax.transAxes, size=12)

    ax = fig.add_subplot(122, axes_class=AxesZero)
    ax.plot(xs, resourceOutput1(xs), "C9", label='stream')
    ax.plot(xs, resourceOutput2(xs), "C8", label='burst')

    for direction in ["left", "bottom"]: #["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")
        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)
        ax.axis[direction].toggle(all=False, label=True)

    for direction in ["right", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    ax.axis((-0.75, 100, -0.075, 10.2))
    ax.set_ylabel('required resources')
    ax.set_xlabel('process\'s progress p')
    ax.legend()
    #ax.text(-0.1, 1.15, '(b)', transform=ax.transAxes, size=12)
    ax.text(-0.1, 1.15, '(b) resources (not storable)', transform=ax.transAxes, size=12)

    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h / 2)

    plt.savefig('figures/samplerequirements.pdf', bbox_inches=Bbox([[0.5, 0], [6, 2.6]]), pad_inches=0.2)
    #plt.show()

if __name__ == "__main__":
    #Test1()
    Test1_Plot()
    MorePaperFigures()
