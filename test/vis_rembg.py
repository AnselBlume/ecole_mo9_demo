# %%
import os
import matplotlib.pyplot as plt
import PIL
from rembg import remove, new_session
from tqdm import tqdm

def plot_fig(orig_img, rem_img):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)
    axes[0].imshow(orig_img)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(rem_img)
    axes[1].axis('off')
    axes[1].set_title('Rembg Image')

    return fig

def get_paths(root_dir: str):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] not in ['.jpg', '.png']:
                print('Filtering file', filename)
                continue

            paths.append(os.path.join(dirpath, filename))

    return paths

# %%
if __name__ == '__main__':
    root_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test'
    out_dir = '/shared/nas2/blume5/fa23/ecole/src/mo9_demo/data/march_demo_test_rembg'

    model_name = 'isnet-general-use'
    close_plots = False

    session = new_session(model_name)
    for image_path in tqdm(get_paths(root_dir)):
        # Process input
        image = PIL.Image.open(image_path).convert('RGB')
        rem_img = remove(image, post_process_mask=True, session=session).convert('RGB')

        # Output
        out_path = os.path.join(
            out_dir,
            os.path.relpath(image_path, root_dir),
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plot = plot_fig(image, rem_img)
        plot.savefig(out_path, bbox_inches='tight')

        if close_plots:
            plt.close(plot)
# %%
