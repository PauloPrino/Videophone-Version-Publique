import os

persons = [
    "Isabelle",
    "Marie",
    "Hugo",
    "Paul",
    "Marc"]

def order_images(folder, person_name):
    i = 0

    # We start by putting a letter followed by a number to have all different to after be able to rename them
    for image in os.listdir(folder):
        if image.endswith('.jpg'):
            os.rename(os.path.join(folder, image), os.path.join(folder, f"{person_name}_temp_{image}"))
        elif image.endswith('.png'):
            os.rename(os.path.join(folder, image), os.path.join(folder, f"{person_name}_temp_{image}"))

    for image in os.listdir(folder):
        i += 1

        if image.endswith('.jpg'):
            image_name = f"{person_name}_{i}.jpg"
            os.rename(os.path.join(folder, image), os.path.join(folder, image_name))
        elif image.endswith('.png'):
            image_name = f"{person_name}_{i}.png"
            os.rename(os.path.join(folder, image), os.path.join(folder, image_name))

for person in persons:
    order_images(f"Labeled_data/{person}", f"{person}")