def modality_dependency(no_text, no_image):
    dependency = {}

    for diag in set(no_text["distribution"]) | set(no_image["distribution"]):
        img_only = diag in no_text["distribution"]
        txt_only = diag in no_image["distribution"]

        if img_only and not txt_only:
            dependency[diag] = "image_dominant"
        elif txt_only and not img_only:
            dependency[diag] = "text_dominant"
        elif img_only and txt_only:
            dependency[diag] = "multimodal"
        else:
            dependency[diag] = "unstable"

    return dependency
