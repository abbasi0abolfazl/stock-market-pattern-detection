import os
import glob
import pandas as pd
import mplfinance as mpf
from ultralyticsplus import YOLO
import cv2

# Load and preprocess data
def load_and_preprocess_data(file_path, num_records=500):
    data = pd.read_csv(file_path).tail(num_records)
    data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    return data

# Generate candlestick charts
def generate_candlestick_charts(data, window_size, save_dir='images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_time_gap = pd.Timedelta(minutes=10)  # Define maximum allowed time gap
    for i in range(0, len(data), window_size):
        ohlc = data.iloc[i:i + window_size]

        # Check for time gaps in the subset data
        ohlc['time_gap'] = ohlc.index.to_series().diff()
        if any(ohlc['time_gap'] > max_time_gap):
            print(f"Skipping chart from index {i} to {i + window_size - 1} due to time gap greater than 10 minutes.")
            continue

        if len(ohlc) < window_size:
            break

        # Generate chart name based on DataFrame length and data range
        start_time = ohlc.index[0].strftime('%Y-%m-%d %H:%M')
        end_time = ohlc.index[-1].strftime('%Y-%m-%d %H:%M')
        chart_name = f'candlestick_chart_{len(data)}_{start_time}_to_{end_time}.png'

        mpf.plot(ohlc, type='candle', style='charles', title=f'Candlestick Chart ({start_time} to {end_time})',
                 ylabel='Price', volume=False, savefig=f'{save_dir}/{chart_name}')

# Detect patterns in images using YOLO and save detected patterns
def detect_patterns_in_images(image_dir, model, save_dir='detected_patterns'):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_paths = glob.glob(f'{image_dir}/*.png')
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = model(image)

        # Only save and show images if patterns are detected
        if len(results[0].boxes) > 0:  # Check if any detections were made
            print(f"Pattern detected in {image_path}")
            annotated_image = results[0].plot()

            # Save the annotated image to the detected_patterns directory
            image_name = os.path.basename(image_path)
            save_path = os.path.join(save_dir, f'detected_{image_name}')
            cv2.imwrite(save_path, annotated_image)
            print(f"Saved detected pattern to {save_path}")

            # Display the annotated image
            cv2.imshow('Detected Pattern', annotated_image)
            cv2.waitKey(0)
        else:
            print(f"No pattern detected in {image_path}")

# Main function
def main():
    # Load and preprocess data
    data = load_and_preprocess_data('XAUUSD_M5.csv')

    # Define window sizes
    window_sizes = [24, 50, 72]  # 2 hours, 4.1 hours, 6 hours

    # Generate candlestick charts for each window size
    for window_size in window_sizes:
        generate_candlestick_charts(data, window_size)

    # Load YOLO model
    model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # Detect patterns in generated images and save detected patterns
    detect_patterns_in_images('images', model)

if __name__ == '__main__':
    main()