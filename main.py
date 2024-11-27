import scanning.visual as vis
import scanning.scanner as sc

matrix = sc.scan('IMG_8165.jpeg')
vis.save(matrix)