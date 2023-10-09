#include <mpi.h>
#include <math.h>

// Define the maximum number of iterations
#define MAX_ITER 100

// Calculate the Mandelbrot value for a given complex number
double mandelbrot(double c_real, double c_imag) {
  double z_real = 0.0;
  double z_imag = 0.0;
  int k;

  for (k = 0; k < MAX_ITER; k++) {
    double temp = z_real * z_real - z_imag * z_imag + c_real;
    z_imag = 2.0 * z_real * z_imag + c_imag;
    z_real = temp;

    if (z_real * z_real + z_imag * z_imag > 4.0) {
      break;
    }
  }

  return k / (double) MAX_ITER;
}

// Static task assignment
void static_task_assignment(int rank, int num_procs, int width, int height, double *data) {
  // Calculate the number of pixels per process
  int pixels_per_proc = width * height / num_procs;

  // Calculate the starting and ending pixel indices for this process
  int start_idx = rank * pixels_per_proc;
  int end_idx = (rank + 1) * pixels_per_proc;

  // Calculate the Mandelbrot value for each pixel in the assigned range
  for (int i = start_idx; i < end_idx; i++) {
    int x = i % width;
    int y = i / width;

    double c_real = xmin + (xmax - xmin) * x / (width - 1);
    double c_imag = ymin + (ymax - ymin) * y / (height - 1);

    data[i] = mandelbrot(c_real, c_imag);
  }
}

// Dynamic task assignment
void dynamic_task_assignment(int rank, int num_procs, int width, int height, double *data) {
  // Create a queue to store the pixels to be calculated
  int queue_size = width * height;
  int *queue = malloc(queue_size * sizeof(int));
  int queue_head = 0;
  int queue_tail = 0;

  // Initialize the queue with all of the pixels
  for (int i = 0; i < queue_size; i++) {
    queue[i] = i;
  }

  // While the queue is not empty, repeatedly poll the queue for work
  while (queue_head != queue_tail) {
    // Get the next pixel from the queue
    int pixel_idx = queue[queue_head++];

    // Calculate the Mandelbrot value for the pixel
    double c_real = xmin + (xmax - xmin) * pixel_idx % width / (width - 1);
    double c_imag = ymin + (ymax - ymin) * pixel_idx / width / (height - 1);

    data[pixel_idx] = mandelbrot(c_real, c_imag);
  }

  // Free the queue
  free(queue);
}

int main(int argc, char **argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the process rank and number of processes
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Check for the correct number of command-line arguments
  if (argc != 5) {
    printf("Usage: %s width height xmin xmax\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Parse the command-line arguments
  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  double xmin = atof(argv[3]);
  double xmax = atof(argv[4]);

  // Allocate memory for the image data
  double *data = malloc(width * height * sizeof(double));

  // Calculate the Mandelbrot set using static task assignment
  double start_time = MPI_Wtime();
  static_task_assignment(rank, num_procs, width, height, data);
  double static_time = MPI_Wtime() - start_time;

  // Calculate the Mandelbrot set using dynamic task assignment
  start_time = MPI_Wtime();
  dynamic_task_assignment(rank, num_procs, width, height, data);
  double dynamic_time = MPI_Wtime() - start_time;

  // Print the time taken by each implementation
  if (rank == 0) {
    printf("Static task assignment time: %f seconds\n", static_time);
    printf("Dynamic task assignment time: %f seconds\n", dynamic_time);
  }

  // Free the image data
  free(data);

  // Finalize MPI
  MPI_Finalize();

  return 0;
}