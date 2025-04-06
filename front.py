import tkinter as tk 
from tkinter import filedialog, messagebox
import prepro 
from prepro import preprocessing
from ocr import perform_ocr

window = tk.Tk()
window.title("Optical Character Recognition")
window.geometry("700x800")

text_output = tk.Text(window, wrap=tk.WORD)
text_output.pack(pady=10, padx=10, expand=True, fill="both")

def selectimage():
    path = filedialog.askopenfilename(filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg"))])
    if path: 
        try:
            preprocessed_path = preprocessing(path)
            text = perform_ocr(preprocessed_path)    # OCR result
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, text)
            messagebox.showinfo("Success", "Text extracted and saved to output.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {e}")

btn = tk.Button(window, text="Select Image for OCR", command=selectimage)
btn.pack(pady=10)

window.mainloop()