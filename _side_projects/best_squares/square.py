import math
import random
import numpy as np

def best_squares(w,h,n_h):
    #print("err", int(h / n_h)-(h / n_h))

    crop = int(h / n_h)
    #print("crop", crop)

    row_list = []
    for i in range(0,n_h):
        #row_list.append([i * crop, i * crop + crop])
        row_list.append([int(i * crop), int(i * crop + crop)])

    n_v = math.ceil(w / crop)
    #print("n_v", n_v)

    loc = (w - crop) / (n_v - 1)
    #print("loc", loc)

    column_list = []

    #print("first",0,crop)
    column_list.append([0,crop])
    for i in range(1, n_v-1):
        #print(i,"ndth", i*loc, i*loc+crop)
        #column_list.append([i*loc, i*loc+crop])
        column_list.append([int(i*loc), int(i*loc+crop)])
    #print("last", w-crop, w)
    column_list.append([w-crop, w])

    print(len(column_list) * len(row_list))

    return column_list, row_list


from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

fig = plt.figure(figsize=(5,5),)
ax = fig.gca()

def redraw(plt, column_list, row_list):
    print("box", column_list[0][1]-column_list[0][0], "x", row_list[0][1]-row_list[0][0])

    plt.axis('equal')
    plt.axis([-10, w + 10, -10, h + 10])
    ax.add_patch(
        patches.Rectangle(
            (0, 0),  # (x,y)
            w,  # width
            h,  # height
            fill=False
        )
    )

    for row in row_list:
        for col in column_list:
            #print("x,", col[0], "y", 0.1, "w", col[1] - col[0], "h", 0.5)

            """
            ax.add_patch(
                patches.Rectangle(
                    (col[0], row[0]),  # (x,y)
                    col[1] - col[0],  # width
                    row[1] - row[0],  # height
                    fill=False
                )
            )
            """

            #jitter = random.uniform(0, 1) * 25
            jitter = 0
            ax.add_patch(
                patches.Rectangle(
                    (col[0] + jitter, row[0] + jitter),
                    col[1] - col[0],
                    row[1] - row[0], fill=False, linewidth=1.0, color=np.random.rand(3, 1)[:, 0]
                )
            )

w = 3840
h = 2160

"""
n = 1
column_list, row_list = best_squares(w, h, n)
redraw(plt, column_list, row_list)
plt.show()
"""


def animate(i):
    print("with",(i+1),"horizontal boxes")
    ax.clear()
    column_list, row_list = best_squares(w, h, i+1)
    redraw(plt, column_list, row_list)
    #ax.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=2000, frames=4, repeat=True)
plt.show()