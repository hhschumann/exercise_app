import streamlit as st
import cv2
import tempfile
import cv2
from pose_estimation.estimation import PoseEstimator
from exercises.squat import Squat
from exercises.hammer_curl import HammerCurl
from exercises.push_up import PushUp
from feedback.layout import layout_indicators
from feedback.information import get_exercise_info
from utils.draw_text_with_background import draw_text_with_background

@st.cache_resource
def get_exercise(selected_exercise):
    if selected_exercise == "hammer_curl":
        return HammerCurl()
    elif selected_exercise == "squat":
        return Squat()
    elif selected_exercise == "push_up":
        return PushUp()
    else:
        print("Invalid exercise type.")
        return


@st.cache_data
def get_annotated_frame(frame, selected_exercise):
    pose_estimator = load_estimator()
    results = pose_estimator.estimate_pose(frame, selected_exercise)
    exercise_info = get_exercise_info(selected_exercise)

    exercise = get_exercise(selected_exercise)

    if results.pose_landmarks:
        if selected_exercise == "squat":
            counter, angle, stage = exercise.track_squat(results.pose_landmarks.landmark, frame)
            layout_indicators(frame, selected_exercise, (counter, angle, stage))
        elif selected_exercise == "hammer_curl":
            (counter_right, angle_right, counter_left, angle_left,
                warning_message_right, warning_message_left, progress_right, progress_left, stage_right, stage_left) = exercise.track_hammer_curl(
                results.pose_landmarks.landmark, frame)
            layout_indicators(frame, selected_exercise,
                                (counter_right, angle_right, counter_left, angle_left,
                                warning_message_right, warning_message_left, progress_right, progress_left, stage_right, stage_left))
        elif selected_exercise == "push_up":
            counter, angle, stage = exercise.track_push_up(results.pose_landmarks.landmark, frame)
            layout_indicators(frame, selected_exercise, (counter, angle, stage))

    draw_text_with_background(frame, f"Exercise: {exercise_info.get('name', 'N/A')}", (40, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255,), (118, 29, 14, 0.79), 1)
    draw_text_with_background(frame, f"Reps: {exercise_info.get('reps', 0)}", (40, 80),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255,), (118, 29, 14, 0.79), 1)
    draw_text_with_background(frame, f"Sets: {exercise_info.get('sets', 0)}", (40, 110),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255,), (118, 29, 14, 0.79),1 )
    return frame

@st.cache_resource
def load_estimator():
    return PoseEstimator()

def process_video(video_path, selected_exercise):
    cap = cv2.VideoCapture(video_path)  
    if not cap.isOpened():
        st.error("Could not open webcam.")

    stop_button = st.button("Stop Demo")  

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        unannotated_frame = frame.copy()
        annotated_frame = get_annotated_frame(frame, selected_exercise)

        org_frame.image(unannotated_frame, channels="BGR")
        ann_frame.image(annotated_frame, channels="BGR")

        if stop_button:
            cap.release()  
            cv2.destroyAllWindows()
            st.stop()  

if __name__=="__main__":
    st.title("AI Workout Assitant")

    selected_exercise = st.sidebar.selectbox(
        "Exercise",
        ("hammer_curl", "squat", "push_up"),
    )

    input_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
    if input_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(input_file.read())
            input_file_path = temp_file.name

    if st.button('Start Demo'):
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()
        process_video(video_path=input_file_path, selected_exercise=selected_exercise)
    
    
        
