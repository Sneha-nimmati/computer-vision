from sim2real_docs.render_docs import get_image_renderings

if __name__ == "__main__":
    input_path = "../data"
    output_path = "../rendered_images"
    background_images = "../background_image"
    get_image_renderings(input_path = input_path,
                    save_path = output_path, bg_images_path = background_images)