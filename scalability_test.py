#!/usr/bin/env python3
"""
Scalability Test Script for Spoof Face Classification System

This script performs load testing on the deployed cloud service by:
1. Simulating multiple concurrent users uploading images
2. Monitoring response times and success rates
3. Incrementally increasing load to identify system limitations
4. Generating performance reports
"""

import os
import sys
import time
import random
import argparse
import requests
import threading
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mimetypes
from tqdm import tqdm
from pathlib import Path


class SpooferLoadTest:
    """Load testing client for the spoof face classification service"""
    
    def __init__(self, base_url, image_dir, results_dir="./results"):
        """
        Initialize the load testing client
        
        Args:
            base_url (str): Base URL of the web service
            image_dir (str): Directory containing test images
            results_dir (str): Directory to save test results
        """
        self.base_url = base_url
        self.image_dir = Path(image_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect all images from the directory
        self.images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.jpeg")) + \
                     list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.gif"))
        if not self.images:
            raise ValueError(f"No valid image files found in {image_dir}")
        print(f"Found {len(self.images)} test images")
        
        # Results storage
        self.results = []
        self.lock = threading.Lock()
        
        # Test status
        self.active_clients = 0
        self.active_clients_lock = threading.Lock()
        self.test_running = False
        
    def check_api_health(self):
        """
        Check if the API is healthy before starting tests
        
        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            # Using the health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200 and response.text == "OK":
                print("API health check passed")
                return True
            else:
                print(f"API health check failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"API health check failed with exception: {e}")
            return False
    
    def test_mongodb_status(self):
        """
        Check MongoDB status
        
        Returns:
            dict: MongoDB status information or None if failed
        """
        try:
            response = requests.get(f"{self.base_url}/mongodb-status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"MongoDB status: {status['status']}")
                return status
            else:
                print(f"Failed to get MongoDB status: {response.status_code}")
                return None
        except Exception as e:
            print(f"Failed to get MongoDB status: {e}")
            return None
    
    def simulate_client(self, client_id, num_requests, delay_range=(1, 3)):
        """
        Simulate a single client making requests to the API
        
        Args:
            client_id (int): ID of the simulated client
            num_requests (int): Number of requests to make
            delay_range (tuple): Range of random delay between requests in seconds
            
        Returns:
            list: Results of the test for this client
        """
        with self.active_clients_lock:
            self.active_clients += 1
        
        client_results = []
        
        for i in range(num_requests):
            if not self.test_running:
                break
                
            # Select a random image
            image_path = random.choice(self.images)
            
            # Read image and prepare request
            try:
                # Get file name and ensure it has a valid extension
                file_name = image_path.name
                
                # Ensure we have the correct MIME type
                content_type = mimetypes.guess_type(str(image_path))[0]
                if not content_type or not content_type.startswith('image/'):
                    content_type = 'image/jpeg'
                
                # Read the image file
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                
                # Track request timing
                start_time = time.time()
                error = None
                response_data = None
                
                try:
                    # Create multipart form data with proper content type
                    files = {'image': (file_name, image_data, content_type)}
                    
                    # Send the request
                    response = requests.post(
                        f"{self.base_url}/upload", 
                        files=files,
                        timeout=30
                    )
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    
                    # Process the response
                    if response.status_code == 200:
                        response_data = response.json()
                        status = "success"
                        num_faces = len(response_data.get('faces', []))
                    else:
                        status = "error"
                        error = f"HTTP {response.status_code}: {response.text}"
                        num_faces = 0
                
                except requests.RequestException as e:
                    response_time = time.time() - start_time
                    status = "error"
                    error = str(e)
                    num_faces = 0
                
                # Record the result
                result = {
                    'client_id': client_id,
                    'request_id': i,
                    'timestamp': time.time(),
                    'image': str(image_path),
                    'status': status,
                    'response_time': response_time,
                    'error': error,
                    'num_faces': num_faces
                }
                
                client_results.append(result)
                
                # Add to global results
                with self.lock:
                    self.results.append(result)
                
                # Simulate delay between requests
                if i < num_requests - 1 and self.test_running:  # No delay after the last request
                    time.sleep(random.uniform(delay_range[0], delay_range[1]))
            
            except Exception as e:
                print(f"Client {client_id} error: {e}")
        
        with self.active_clients_lock:
            self.active_clients -= 1
            
        return client_results
    
    def run_load_test(self, num_clients, requests_per_client, ramp_up=False):
        """
        Run a load test with multiple simulated clients
        
        Args:
            num_clients (int): Number of concurrent clients to simulate
            requests_per_client (int): Number of requests each client should make
            ramp_up (bool): If True, gradually add clients instead of all at once
            
        Returns:
            pd.DataFrame: Results of the test
        """
        print(f"Starting load test with {num_clients} clients, {requests_per_client} requests per client")
        
        # Check if API is healthy first
        if not self.check_api_health():
            print("API health check failed. Not running load test.")
            return None
            
        # Reset results
        self.results = []
        self.active_clients = 0
        self.test_running = True
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            
            # Submit client jobs
            for i in range(num_clients):
                if ramp_up:
                    # Add 1 client every 2 seconds if ramping up
                    if i > 0:
                        time.sleep(2)
                
                if not self.test_running:
                    break
                    
                future = executor.submit(
                    self.simulate_client,
                    client_id=i,
                    num_requests=requests_per_client
                )
                futures.append(future)
                print(f"Started client {i}")
            
            # Create progress bar
            total_requests = num_clients * requests_per_client
            with tqdm(total=total_requests, desc="Progress") as pbar:
                previous_count = 0
                
                # Update the progress bar until all clients are done
                while self.active_clients > 0 or not all(f.done() for f in futures):
                    with self.lock:
                        current_count = len(self.results)
                    
                    # Update progress bar with new results
                    if current_count > previous_count:
                        pbar.update(current_count - previous_count)
                        previous_count = current_count
                    
                    time.sleep(0.5)
                
                # Final update to the progress bar
                with self.lock:
                    current_count = len(self.results)
                if current_count > previous_count:
                    pbar.update(current_count - previous_count)
                
        end_time = time.time()
        duration = end_time - start_time
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        success_rate = (df['status'] == 'success').mean() * 100
        mean_response_time = df['response_time'].mean()
        p95_response_time = df['response_time'].quantile(0.95)
        p99_response_time = df['response_time'].quantile(0.99)
        requests_per_second = len(df) / duration
        
        print(f"\nLoad Test Summary:")
        print(f"  Total Duration: {duration:.2f} seconds")
        print(f"  Success Rate: {success_rate:.2f}%")
        print(f"  Mean Response Time: {mean_response_time:.2f} seconds")
        print(f"  95th Percentile Response Time: {p95_response_time:.2f} seconds")
        print(f"  99th Percentile Response Time: {p99_response_time:.2f} seconds")
        print(f"  Throughput: {requests_per_second:.2f} requests/second")
        
        # Save results to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(self.results_dir / f"load_test_{timestamp}_{num_clients}_clients.csv", index=False)
        
        # Generate visualizations
        self.generate_visualizations(df, timestamp, num_clients)
        
        return df
    
    def generate_visualizations(self, df, timestamp, num_clients):
        """
        Generate visualizations of the load test results
        
        Args:
            df (pd.DataFrame): Load test results
            timestamp (str): Timestamp string for filenames
            num_clients (int): Number of clients used in the test
        """
        # Plot response times over time
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'] - df['timestamp'].min(), df['response_time'], 'o-', alpha=0.3)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Response Time (seconds)')
        plt.title(f'Response Times Over Time ({num_clients} clients)')
        plt.grid(True)
        
        # Add a trendline
        z = np.polyfit(df['timestamp'] - df['timestamp'].min(), df['response_time'], 1)
        p = np.poly1d(z)
        plt.plot(df['timestamp'] - df['timestamp'].min(), p(df['timestamp'] - df['timestamp'].min()), 
                "r--", linewidth=2)
        
        plt.savefig(self.results_dir / f"response_times_{timestamp}.png")
        
        # Plot response time histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['response_time'], bins=30, alpha=0.7)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Count')
        plt.title(f'Response Time Distribution ({num_clients} clients)')
        plt.grid(True)
        plt.savefig(self.results_dir / f"response_hist_{timestamp}.png")
        
        # Plot success rate by client
        success_by_client = df.groupby('client_id')['status'].apply(
            lambda x: (x == 'success').mean() * 100).reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(success_by_client['client_id'], success_by_client['status'])
        plt.xlabel('Client ID')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Success Rate by Client ({num_clients} clients)')
        plt.grid(True)
        plt.savefig(self.results_dir / f"success_rate_{timestamp}.png")
        
        # If there are any errors, create an error rate chart over time
        if (df['status'] == 'error').any():
            # Group by minute and calculate error rate
            df['minute'] = (df['timestamp'] - df['timestamp'].min()) // 60
            error_rate = df.groupby('minute')['status'].apply(
                lambda x: (x == 'error').mean() * 100).reset_index()
            
            plt.figure(figsize=(12, 6))
            plt.bar(error_rate['minute'], error_rate['status'])
            plt.xlabel('Time (minutes)')
            plt.ylabel('Error Rate (%)')
            plt.title(f'Error Rate Over Time ({num_clients} clients)')
            plt.grid(True)
            plt.savefig(self.results_dir / f"error_rate_{timestamp}.png")
    
    def run_scalability_test(self, start_clients=1, max_clients=100, step=10, 
                           requests_per_client=5, threshold_pct=80):
        """
        Run a scalability test by incrementally increasing the number of clients
        until performance degrades
        
        Args:
            start_clients (int): Starting number of concurrent clients
            max_clients (int): Maximum number of concurrent clients
            step (int): Increment in number of clients per test
            requests_per_client (int): Requests per client in each test
            threshold_pct (int): Success rate threshold percentage to stop test
            
        Returns:
            pd.DataFrame: Summary of all test runs
        """
        print(f"Starting scalability test from {start_clients} to {max_clients} clients")
        
        # Storage for test results summary
        summary = []
        
        # Run tests with increasing client count
        clients = start_clients
        while clients <= max_clients:
            print(f"\n--- Testing with {clients} concurrent clients ---")
            
            # Run the load test
            df = self.run_load_test(clients, requests_per_client)
            
            if df is None or len(df) == 0:
                print(f"Test failed with {clients} clients")
                break
                
            # Calculate summary statistics
            success_rate = (df['status'] == 'success').mean() * 100
            mean_response_time = df['response_time'].mean()
            p95_response_time = df['response_time'].quantile(0.95)
            requests_per_second = len(df) / (df['timestamp'].max() - df['timestamp'].min())
            
            # Add to summary
            summary.append({
                'num_clients': clients,
                'success_rate': success_rate,
                'mean_response_time': mean_response_time,
                'p95_response_time': p95_response_time,
                'requests_per_second': requests_per_second
            })
            
            # Check if performance has degraded below threshold
            if success_rate < threshold_pct:
                print(f"Success rate ({success_rate:.2f}%) fell below threshold ({threshold_pct}%)")
                print(f"Maximum reliable concurrency: {clients - step} clients")
                break
                
            # Check if response times are becoming unreasonable (over 10 seconds)
            if mean_response_time > 10:
                print(f"Mean response time ({mean_response_time:.2f}s) exceeded 10 seconds")
                print(f"Maximum responsive concurrency: {clients - step} clients")
                break
                
            # Increment client count
            clients += step
            
            # Small delay between tests to let system recover
            time.sleep(5)
        
        # Convert summary to DataFrame and save
        summary_df = pd.DataFrame(summary)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        summary_df.to_csv(self.results_dir / f"scalability_test_{timestamp}.csv", index=False)
        
        # Generate summary visualization
        self.generate_scalability_visualization(summary_df, timestamp)
        
        return summary_df
    
    def generate_scalability_visualization(self, df, timestamp):
        """
        Generate visualizations of the scalability test results
        
        Args:
            df (pd.DataFrame): Scalability test summary results
            timestamp (str): Timestamp string for filenames
        """
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot success rate
        color = 'tab:blue'
        ax1.set_xlabel('Number of Concurrent Clients')
        ax1.set_ylabel('Success Rate (%)', color=color)
        ax1.plot(df['num_clients'], df['success_rate'], 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Response Time (seconds)', color=color)
        ax2.plot(df['num_clients'], df['mean_response_time'], 's-', color=color, label='Mean')
        ax2.plot(df['num_clients'], df['p95_response_time'], '^-', color='tab:orange', label='95th Percentile')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add throughput as a third line (on first y-axis, but different scale)
        color = 'tab:green'
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1)) 
        ax3.set_ylabel('Throughput (requests/second)', color=color)
        ax3.plot(df['num_clients'], df['requests_per_second'], 'd-', color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        
        # Add legend and title
        fig.tight_layout()
        fig.suptitle('System Scalability Test Results', fontsize=16)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2 + lines3, ['Success Rate'] + labels2 + ['Throughput'], 
                   loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
        
        plt.savefig(self.results_dir / f"scalability_results_{timestamp}.png", bbox_inches='tight')
        
        # Create an additional chart showing relationship between throughput and response time
        plt.figure(figsize=(10, 6))
        plt.scatter(df['requests_per_second'], df['mean_response_time'], 
                   s=df['num_clients']*5, alpha=0.7)
        
        # Add client count labels to each point
        for i, row in df.iterrows():
            plt.annotate(f"{int(row['num_clients'])} clients", 
                        (row['requests_per_second'], row['mean_response_time']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Throughput (requests/second)')
        plt.ylabel('Mean Response Time (seconds)')
        plt.title('Response Time vs. Throughput')
        plt.grid(True)
        plt.savefig(self.results_dir / f"throughput_vs_response_{timestamp}.png")


def main():
    """Main function to run the load testing"""
    parser = argparse.ArgumentParser(description='Load test the spoof detection service')
    parser.add_argument('--url', type=str, default='http://34.10.15.184',
                      help='Base URL of the service')
    parser.add_argument('--image-dir', type=str, required=True,
                      help='Directory containing test images')
    parser.add_argument('--mode', type=str, choices=['load', 'scalability'], default='scalability',
                      help='Test mode: single load test or scalability test')
    parser.add_argument('--clients', type=int, default=10,
                      help='Number of concurrent clients (for load test mode)')
    parser.add_argument('--requests', type=int, default=5,
                      help='Number of requests per client')
    parser.add_argument('--max-clients', type=int, default=100,
                      help='Maximum number of clients (for scalability test)')
    parser.add_argument('--step', type=int, default=10,
                      help='Step size for increasing clients (for scalability test)')
    parser.add_argument('--results-dir', type=str, default='./results',
                      help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Create the load tester
    load_tester = SpooferLoadTest(args.url, args.image_dir, args.results_dir)
    
    try:
        # Run test based on mode
        if args.mode == 'load':
            load_tester.run_load_test(args.clients, args.requests)
        else:  # scalability
            load_tester.run_scalability_test(
                start_clients=args.clients,
                max_clients=args.max_clients,
                step=args.step,
                requests_per_client=args.requests
            )
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        load_tester.test_running = False
    except Exception as e:
        print(f"Error during test: {e}")
    
    print("Load testing complete")


if __name__ == "__main__":
    main()