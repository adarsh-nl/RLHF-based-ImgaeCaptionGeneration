
#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("/content/drive/MyDrive/CV Project - Image caption generator/Models/tokenizer.p","rb"))

# Register the custom loss function
with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
    # Load the model
    model = tf.keras.models.load_model('/content/models/model_rl.h5')

xception_model = Xception(include_top=False, pooling="avg")

img_path = "/content/test_image4.png"
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print("Caption 1: ", description)

plt.imshow(img)
