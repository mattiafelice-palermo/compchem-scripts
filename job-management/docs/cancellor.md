# Cancellor 🎩 - Easy deletion of PBS/SLURM jobs

Greetings. Cancellor is here to help 🧐.

In the grand tradition of elegant politicians managing affairs with grace and decisiveness, Cancellor gracefully handles and terminates jobs on computational clusters using PBS and SLURM schedulers. Whether you need to clear the queue for a fresh start or selectively cancel tasks, Cancellor performs the task with the finesse of an old-time chancellor.

## Overview

Cancellor is a Python-based command-line utility designed to manage and cancel jobs in a PBS or SLURM managed cluster environment. With a simple yet powerful interface, Cancellor allows for flexible job selection options, including batch operations, with an underlying robustness to ensure jobs are indeed canceled, even if it takes multiple attempts.

## Installation

To get started with Cancellor, clone this repository or download the `cancellor.py` script directly. Ensure that Python 3 is installed on your system, and make the script executable:

```bash
chmod +x cancellor.py
```

## Usage

Cancellor greets you upon execution and offers various options to manage your jobs. Here are a few ways to use Cancellor:

### Basic Usage

Simply run Cancellor to interactively list and choose jobs to cancel:

```bash
./cancellor.py
```

### Cancel All Jobs

To cancel all jobs (with a prompt for confirmation), use the `--all` flag:

```bash
./cancellor.py --all
```

### Specify Job Numbers

To cancel specific jobs by their queue numbers (e.g., 0, 1-3, 5), use the `--which` option:

```bash
./cancellor.py --which "0, 1-3, 5"
```

### Advanced Options

- `--attempts`: Specify the number of attempts to cancel a job if the first try fails.
- `--delay`: Set how long to wait (in seconds) before reattempting a cancellation.
- `--scheduler`: Manually specify the scheduler if auto-detection is not desired.

### Example with Advanced Options

```bash
./cancellor.py --which "1, 2-4" --attempts 3 --delay 2
```

## Farewell

Upon completion, Cancellor bids you "Greetings and au revoir 🎩", having hopefully cleared the path for your next computational endeavors.

For more information on command-line options, run `./cancellor.py -h`.

---

Cancellor - Elegantly managing the end of computational tasks, one job at a time.
