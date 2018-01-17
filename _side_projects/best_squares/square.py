import math

def best_squares(w,h,n_h):
    print("err", int(h / n_h)-(h / n_h))

    crop = int(h / n_h)
    print("crop", crop)

    row_list = []
    for i in range(0,n_h):
        row_list.append([i * crop, i * crop + crop])

    areas = math.ceil(w / crop)
    print("areas", areas)

    loc = (w - crop) / (areas - 1)
    print("loc", loc)

    column_list = []

    print("first",0,crop)
    column_list.append([0,crop])
    for i in range(1, areas-1):
        print(i,"ndth", i*loc, i*loc+crop)
        column_list.append([i*loc, i*loc+crop])
    print("last", w-crop, w)
    column_list.append([w-crop, w])
    return column_list, row_list

w = 1024
h = 620
column_list, row_list = best_squares(w,h,2)

print(column_list)
print(row_list)



import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(5,5),)
plt.axis('equal')
plt.axis([-10, w+10, -10, h+10])

ax = plt.gca()
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
        print("x,",col[0], "y",0.1, "w", col[1]-col[0], "h", 0.5)

        ax.add_patch(
            patches.Rectangle(
                (col[0], row[0]),  # (x,y)
                col[1] - col[0],  # width
                row[1] - row[0],  # height
                fill=False
            )
        )

plt.rcParams["figure.figsize"] = [100,100]

plt.show()

