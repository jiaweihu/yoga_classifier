# https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR#scrollTo=bERVPO8Ja6j7

from BootstrapHelper import BootstrapHelper

# Required structure of the images_in_folder:
#
#   fitness_poses_images_in/
#     pushups_up/
#       image_001.jpg
#       image_002.jpg
#       ...
#     pushups_down/
#       image_001.jpg
#       image_002.jpg
#       ...
#     ...

bootstrap_images_in_folder = 'data\yoga\surya\in'
bootstrap_images_out_folder = 'data\yoga\surya\out'
bootstrap_csvs_out_folder = 'data\yoga\surya\csv'

# Initialize helper.
bootstrap_helper = BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=bootstrap_csvs_out_folder,
)

# Check how many pose classes and images for them are available.
bootstrap_helper.print_images_in_statistics()

# Bootstrap all images.
# Set limit to some small number for debug.
bootstrap_helper.bootstrap(per_pose_class_limit=None)

# Check how many images were bootstrapped.
bootstrap_helper.print_images_out_statistics()

# After initial bootstrapping images without detected poses were still saved in
# the folderd (but not in the CSVs) for debug purpose. Let's remove them.
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()



