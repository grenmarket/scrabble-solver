import scanning.visual as vis
import scanning.scanner as sc

matrix = sc.scan('/home/s/Downloads/test-smooth.jpg')
vis.save(matrix)