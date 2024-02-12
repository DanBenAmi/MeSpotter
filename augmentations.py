import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class RandomRotateCrop(object):
    def __init__(self, max_angle):
        """
        Args:
            max_angle (float): Maximum rotation angle in degrees.
        """
        self.max_angle = max_angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated and cropped.

        Returns:
            PIL Image: Rotated and cropped image.
        """
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        img_np = np.array(img)
        rotated_img_np = self.rotate_image(img_np, angle)
        image_height, image_width = img_np.shape[0:2]
        # new_width, new_height = largest_rotated_rect(image_width, image_height, math.radians(angle))
        new_width, new_height = self.rotatedRectWithMaxArea(image_width, image_height, math.radians(angle))
        cropped_img_np = self.crop_around_center(rotated_img_np, int(new_width), int(new_height))

        # Convert back to PIL Image
        img_pil = Image.fromarray(cropped_img_np)
        return img_pil

    def rotatedRectWithMaxArea(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return wr, hr

    def rotate_image(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def demo():
        """
        Demos the largest_rotated_rect function
        """

        image = cv2.imread("lenna_rectangle.png")
        image_height, image_width = image.shape[0:2]

        cv2.imshow("Original Image", image)

        print
        "Press [enter] to begin the demo"
        print
        "Press [q] or Escape to quit"

        key = cv2.waitKey(0)
        if key == ord("q") or key == 27:
            exit()

        for i in np.arange(0, 360, 0.5):
            image_orig = np.copy(image)
            image_rotated = rotate_image(image, i)
            image_rotated_cropped = crop_around_center(
                image_rotated,
                *largest_rotated_rect(
                    image_width,
                    image_height,
                    math.radians(i)
                )
            )

            key = cv2.waitKey(2)
            if (key == ord("q") or key == 27):
                exit()

            cv2.imshow("Original Image", image_orig)
            cv2.imshow("Rotated Image", image_rotated)
            cv2.imshow("Cropped Image", image_rotated_cropped)

        print
        "Done"


def get_augmentations(augmentations):
    # Define a dictionary mapping augmentation names to their corresponding torchvision transforms
    all_augmentations = {
        "random_resize_crop": transforms.RandomResizedCrop(size=(160,112), scale=(0.9, 1.0), ratio=(0.75, 1),
                                                           interpolation=InterpolationMode.BILINEAR),
        "random_color_jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        "random_horizontal_flip": transforms.RandomHorizontalFlip(p=0.5),
        "random_blur": transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        "random_rotate_crop": RandomRotateCrop(max_angle=80)
        # Add more standard augmentations for faces here
    }

    if augmentations == "all":
        # Use all augmentations
        transform_list = list(all_augmentations.values())
    elif isinstance(augmentations, list):
        # Use only the specified augmentations
        transform_list = [all_augmentations[aug] for aug in augmentations if aug in all_augmentations]
    else:
        raise ValueError("augmentations must be 'all' or a list of augmentation names")

    # Always include ToTensor() as the last transformation
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


if __name__ == "__main__":
    img_path = "DataBase/FaceDB/1701271022209_1.jpg"
    image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
    obj = RandomRotateCrop(90)
    for i in range(5):
        rot_img = obj(image)
        plt.imshow(rot_img)
        plt.show(block=True)
