def save_status(status_file, status):
  if status_file is not None and status is not None:
    with open(status_file, "a+") as out_file:
      out_file.write(str(status) + "\n")
