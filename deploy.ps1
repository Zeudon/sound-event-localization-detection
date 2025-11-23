# Deployment Script
$User = "nd2746"
$HostName = "greene.hpc.nyu.edu"
$RemoteDir = "/scratch/$User/seld_project"

# Files to transfer
$Files = @(
    "config.py",
    "dataset.py",
    "loss.py",
    "main.py",
    "model.py",
    "model_conformer.py",
    "model_crnn.py",
    "trainer.py",
    "utils.py",
    "visualization.py",
    "run_job.slurm"
)

# Join files with spaces for single SCP command
$FileList = $Files -join " "

Write-Host "Creating remote directory..."
# -o StrictHostKeyChecking=no avoids the error when hitting different login nodes
ssh -o StrictHostKeyChecking=no $User@$HostName "mkdir -p $RemoteDir"

Write-Host "Transferring files..."
# Transfer all files in one go to be faster and avoid repeated auth checks
# We use Invoke-Expression to handle the variable expansion correctly in PowerShell
$SCPCommand = "scp -o StrictHostKeyChecking=no $FileList ${User}@${HostName}:${RemoteDir}/"
Invoke-Expression $SCPCommand

Write-Host "Deployment complete!"
Write-Host "To run the job, SSH into the HPC and run: sbatch run_job.slurm"
