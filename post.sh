# get command line arguments

if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    message="Update"
  else
    message=$1
fi
hugo --minify
git add -A
git commit -m "$message"
git push

