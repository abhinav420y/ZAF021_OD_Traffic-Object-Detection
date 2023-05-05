from moviepy.editor import VideoFileClip
from IPython.display import HTML

sess = K.get_session()

def predict(sess, image_file):
    
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes,classes],
                                                  feed_dict={yolo_model.input:image_data , K.learning_phase(): 0})


    colors = generate_colors(class_names)
    
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    image.save(os.path.join("./Dense_LISA_1/Dense/frames/output/temp.jpg"), quality=90)
    output_image = plt.imread(os.path.join("./Dense_LISA_1/Dense/frames/output/temp.jpg"))

    return output_image


def image_pipeline(image):
    output_image = predict(sess, image)
    return output_image

video_output = './Dense_Output.mp4'
# clip1 = VideoFileClip("./Dense_LISA_1/Dense/jan28.avi").subclip(0,2)
clip1 = VideoFileClip("./Dense_LISA_1/Dense/jan28.avi")
video_clip = clip1.fl_image(image_pipeline) #NOTE: this function expects color images!!
%time video_clip.write_videofile(video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))