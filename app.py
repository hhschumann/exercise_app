import streamlit as st
import cv2
import tempfile
import cv2
import time
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

@st.cache_data
def get_annotated_frames(video_path, selected_exercise):
    cap = cv2.VideoCapture(video_path)  
    if not cap.isOpened():
        st.error("Could not open webcam.")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames=[]
    annotated_frames=[]

    count=0
    while count < n_frames:
        success, frame = cap.read()
        if not success: break
        org_frame = frame.copy()
        annotated_frame = get_annotated_frame(frame, selected_exercise)
        original_frames.append(org_frame)
        annotated_frames.append(annotated_frame) 
        percent_complete = int(100*((count+1)/n_frames))
        count+=1
    #cap.release()  
    # cv2.destroyAllWindows()
    return original_frames, annotated_frames, fps

@st.cache_resource
def load_estimator():
    return PoseEstimator()

def process_video_sequenced(video_path, selected_exercise):
    original_frames, annotated_frames, fps = get_annotated_frames(video_path, selected_exercise)
    stop_button = st.button("Stop Demo")  
    for i in range(len(annotated_frames)-1):
        org_frame.image(original_frames[i], channels="BGR")
        ann_frame.image(annotated_frames[i], channels="BGR")
        time.sleep(1/fps)
        if stop_button:
            st.stop()  

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

    if 'original_frames' not in st.session_state:
        st.session_state['original_frames'] = []
    if 'annotated_frames' not in st.session_state:
        st.session_state['annotated_frames'] = []
    if 'fps' not in st.session_state:
        st.session_state['fps'] = 30  # Default value

    def resize_frame(frame, width=640):
        height = int(frame.shape[0] * (width / frame.shape[1]))
        return cv2.resize(frame, (width, height))

    if st.button('Process Video'):
        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            st.error("Could not open video file.")
            st.stop()

        st.session_state['fps'] = int(cap.get(cv2.CAP_PROP_FPS)) // 2  # Reduce FPS
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.session_state['original_frames'] = []
        st.session_state['annotated_frames'] = []

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame)  # Resize for better performance

            unannotated_frame = frame.copy()
            annotated_frame = get_annotated_frame(frame, selected_exercise)

            st.session_state['original_frames'].append(unannotated_frame)
            st.session_state['annotated_frames'].append(annotated_frame)

            count += 1
            if count >= n_frames - 1:
                break

        cap.release()
        cv2.destroyAllWindows()

    import threading

    def run_video_playback():
        col1, col2 = st.columns(2)
        org_frame_container = col1.empty()
        ann_frame_container = col2.empty()

        for i in range(len(st.session_state['annotated_frames'])):
            org_frame_container.image(st.session_state['original_frames'][i], channels="BGR")
            ann_frame_container.image(st.session_state['annotated_frames'][i], channels="BGR")
            time.sleep(1 / st.session_state['fps'])

    if st.button('Run Model'):
        playback_thread = threading.Thread(target=run_video_playback)
        playback_thread.start()


