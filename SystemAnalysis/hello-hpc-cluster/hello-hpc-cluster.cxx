#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include <sstream>

using namespace std;

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

#define MAXLEN_HOSTNAME 127 // @TODO: Determine this length more "canonically" using `sysconf()`
  char hostname[MAXLEN_HOSTNAME+1];
  gethostname(hostname, MAXLEN_HOSTNAME);

  bool erase=false;
  int index=0;
  while(hostname[index]!='\0')
  {
    if(hostname[index]=='.')
      erase=true;
    if(erase)
      hostname[index]='\0';
    index++;
  }

  int num_threads;
#pragma omp parallel
#pragma omp single
  num_threads = omp_get_num_threads();

  if(rank == 0) {
    cerr << "=== Job:"
         << " (" << num_procs << " processes)"
         << " x (" << num_threads << " threads per process)"
         << " == " << (num_procs * num_threads) << "-way concurrency"
         << " ===" << endl;

    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING+1];
    int mpi_version_len = 0;
    MPI_Get_library_version(mpi_version, &mpi_version_len);
    cerr << endl
         << "MPI version [" << mpi_version_len << "/" << MPI_MAX_LIBRARY_VERSION_STRING << " chars]:" << endl
         << mpi_version
         << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

#pragma omp parallel
  {
#pragma omp critical
    {
      int thread_id = omp_get_thread_num();
      int cpu = sched_getcpu();
      stringstream msg;
      msg << "[" << hostname << ":p" << rank << "/" << num_procs << "::t" << thread_id << "/" << num_threads << "::c" << cpu << "] Hello, world!" << endl;
      cerr << msg.str();
    }
  }

  MPI_Finalize();
  return 0;
}
