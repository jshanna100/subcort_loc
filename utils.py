import numpy as np
import matplotlib.pyplot as plt

def locate_vertex(vtx, labels):
    for lab_idx, lab in enumerate(labels):
        if vtx in lab.vertices:
            return lab
    return None

def make_brain_image(views, brain, orient="horizontal", text="",
                     text_loc=None, text_pan=None, fontsize=160,
                     legend=None, legend_pan=None, cbar=None,
                     vmin=None, vmax=None, cbar_label=""):
    img_list = []
    axis = 1 if orient=="horizontal" else 0
    for k,v in views.items():
        brain.show_view(**v)
        scr = brain.screenshot()
        img_list.append(scr)
    if text != "":
        img_txt_list = []
        brain.add_text(0, 0.8, text, text_loc, font_size=fontsize, color=(1,1,1))
        for k,v in views.items():
            brain.show_view(**v)
            scr = brain.screenshot()
            img_txt_list.append(scr)
        img_list[text_pan] = img_txt_list[text_pan]
    if legend:
        legend_list = []
        leg = brain._renderer.plotter.add_legend(legend, bcolor="w")
        for k,v in views.items():
            brain.show_view(**v)
            scr = brain.screenshot()
            legend_list.append(scr)
        img_list[legend_pan] = legend_list[legend_pan]
    if orient == "square": # only works for 2x2
        h, w, _ = img_list[0].shape
        img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        img[:h, :w, ] = img_list[0]
        img[:h, w:, ] = img_list[1]
        img[h:, :w, ] = img_list[2]
        img[h:, w:, ] = img_list[3]
    else:
        img = np.concatenate(img_list, axis=axis)

    if cbar:
        norm = Normalize(vmin, vmax)
        scalmap = cm.ScalarMappable(norm, cbar)
        if orient == "horizontal":
            colbar_size = (img_list[-1].shape[0]*4, img_list[-1].shape[1]/6)
        else:
            colbar_size = (img_list[-1].shape[0]/6, img_list[-1].shape[1]*4)
        colbar_size = np.array(colbar_size) / 100
        fig, ax = plt.subplots(1,1, figsize=colbar_size)
        colbar = plt.colorbar(scalmap, cax=ax, orientation=orient)
        ax.tick_params(labelsize=48)
        if orient == "horizontal":
            ax.set_xlabel(cbar_label, fontsize=48)
        else:
            ax.set_ylabel(cbar_label, fontsize=48)
        fig.tight_layout()
        fig.canvas.draw()
        mat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        mat = mat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.concatenate((img, mat), axis=abs(axis-1))

    if legend:
        brain._renderer.plotter.remove_legend()

    return img
