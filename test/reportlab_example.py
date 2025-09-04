#!/usr/bin/env python



img = BytesIO()

fig, ax = plt.subplots(figsize=(canvaswidth, canvaswidth))

ax.plot([1,2,3],[6,5,4],antialiased=True,linewidth=2,color='red',label='a curve')

fig.savefig(img,format='PDF')

return(PdfImage(img))