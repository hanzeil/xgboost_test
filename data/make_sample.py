import os
import random


def make_sample():
	file = open("adult.data.csv")
	lines = file.readlines(2000000)
	print len(lines)
	sample_lines = []
	i, j = 0, 0
	max_num = 7500
	while i < max_num or j < max_num:
		r = random.randint(0, len(lines) - 1)
		if lines[r][-3] == '0':
			if i < max_num:
				sample_lines.append(lines[r])
				i += 1
				del lines[r]
			else:
				continue
		else:
			if j < max_num:
				sample_lines.append(lines[r])
				j += 1
				del lines[r]
			else:
				continue

	out_file1 = open("adult1.data.csv", 'w')
	out_file2 = open("adult2.data.csv", 'w')
	for line in lines:
		out_file1.write(line)
	for line in sample_lines:
		out_file2.write(line)


def make_sample2():
	file = open("adult2.data.csv")
	lines = file.readlines(2000000)
	print len(lines)
	sample_lines = []
	i, j = 0, 0
	max_num = 3000
	while i < 3000:
		r = random.randint(0, len(lines) - 1)
		sample_lines.append(lines[r])
		i += 1
		del lines[r]
	out_file1 = open("adult3.data.csv", 'w')
	out_file2 = open("adult4.data.csv", 'w')
	for line in lines:
		out_file1.write(line)
	for line in sample_lines:
		out_file2.write(line)


if __name__ == '__main__':
	# make_sample()
	make_sample2()
	pass
