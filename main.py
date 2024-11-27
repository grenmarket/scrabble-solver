import scanning.visual as vis
import scanning.scanner as sc

matrix = sc.scan('/home/s/Downloads/IMG_8165.jpeg')
vis.save(matrix)