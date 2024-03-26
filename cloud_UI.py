import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
from mmengine.model import revert_sync_batchnorm
from argparse import ArgumentParser
from mmseg.apis import inference_model, init_model
import numpy as np
import os
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=1, help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    return args


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("云检测系统")
        self.root.geometry("1000x500")

        self.input_image = None
        self.input_image_path = ""
        self.segmented_image = None
        self.output_image_path = ""
        self.mix_image = None
        self.canvas_image = None
        self.canvas_segmented = None

        self.overlap="1.00"
        self.overlap2="1.00"

        # 创建界面元素
        self.create_widgets()


    def create_widgets(self):
        # 第一列，创建输入图片展示区域
        self.input_image_label = tk.Label(self.root, text="输入图像",font=("TkDefaultFont", 16))
        self.input_image_label.place(x=25, y=15, width=350,height=30)

        self.input_image_panel = tk.Label(self.root,relief=GROOVE)
        self.input_image_panel.place(x=25, y=60, width=350, height=350)

        self.load_image_button = tk.Button(self.root, text="上传", command=self.load_image_button,font=("TkDefaultFont", 12))
        self.load_image_button.place(x=25, y=425, width=50, height=30)

        self.image_path_entry = tk.Entry(self.root)
        self.image_path_entry.place(x=75, y=425, width=300, height=30)
        self.image_path_entry.bind("<Return>", self.load_image_entry)

        # #第二列，参数和开始分割按钮、推理速度和含云量展示区域
        self.start_segmentation_button = tk.Button(self.root, text="开始检测", command=self.start_segmentation,
                                                   font=('华文新魏', 18))
        self.start_segmentation_button.place(x=440, y=140, width=120, height=120)

        self.class_ratio_label1 = tk.Label(self.root, text="图像大小：",font=("TkDefaultFont", 12))
        self.class_ratio_label1.place(x=400, y=280, width=77, height=30)

        self.class_ratio_panel1 = tk.Label(self.root,relief=GROOVE,font=("TkDefaultFont", 12))
        self.class_ratio_panel1.place(x=477, y=280, width=125, height=30)

        self.class_ratio_label2 = tk.Label(self.root, text="推理时间：",font=("TkDefaultFont", 12))
        self.class_ratio_label2.place(x=400, y=330, width=77, height=30)

        self.class_ratio_panel2 = tk.Label(self.root,relief=GROOVE,font=("TkDefaultFont", 12))
        self.class_ratio_panel2.place(x=477, y=330, width=125, height=30)

        self.class_ratio_label3 = tk.Label(self.root, text="含云量：",font=("TkDefaultFont", 12))
        self.class_ratio_label3.place(x=400, y=380, width=77, height=30)

        self.class_ratio_panel3 = tk.Label(self.root,relief=GROOVE,font=("TkDefaultFont", 12))
        self.class_ratio_panel3.place(x=477, y=380, width=125, height=30)
        #
        # #第三列，创建输出分割结果展示区域
        self.output_image_label = tk.Label(self.root, text="检测结果",font=("TkDefaultFont", 16))
        self.output_image_label.place(x=625, y=15, width=350,height=30)

        self.output_image_panel = tk.Label(self.root, relief=GROOVE)
        self.output_image_panel.place(x=625, y=60, width=350, height=350)
        self.output_image_panel.bind("<Double-Button-1>", self.create_canvas_window)

        self.slider = tk.Scale(self.root, from_=0.00, to=1.00, orient=tk.HORIZONTAL, command=self.update_blend,
                               resolution=0.05)
        self.slider.set(1.00)
        self.slider.config(state=DISABLED)
        self.slider.place(x=625, y=410, width=300, height=45)

        self.save_image_button = tk.Button(self.root, text="保存", command=self.save_image,font=("TkDefaultFont", 12))
        self.save_image_button.place(x=925, y=425, width=50, height=30)

    def show_image(self):
        # 显示选择的图片
        self.input_image = Image.open(self.input_image_path)
        image_size_text = f"{self.input_image.size[0]}×{self.input_image.size[1]}"
        input_show=self.input_image.resize((350, 350))  # 调整图片大小以适应窗口
        input_im = ImageTk.PhotoImage(input_show)
        self.input_image_panel.config(image=input_im)
        self.input_image_panel.image = input_im

        #将各个显示的窗口重置
        self.class_ratio_panel1.config(text=image_size_text)
        self.class_ratio_panel2.config(text="")
        self.class_ratio_panel3.config(text="")

        #设置滑块
        self.slider.set(1.00)
        self.slider.config(state=DISABLED)

        #清空输出窗口图片
        self.output_image_panel.config(image=None)
        self.output_image_panel.image = None

    def load_image_button(self):
        # 从本地文件夹中加载图片
        self.input_image_path = filedialog.askopenfilename()
        self.image_path_entry.delete(0, tk.END)  # 清空文本框内容
        self.image_path_entry.insert(0, self.input_image_path)
        self.show_image()

    def load_image_entry(self,event):
        # 从文本框中获取文件路径
        self.input_image_path = self.image_path_entry.get()

        self.image_path_entry.delete(0, tk.END)  # 清空文本框内容
        self.image_path_entry.insert(0, self.input_image_path)

        self.show_image()

    def save_image(self):
        #保存图片，默认是读取图片路径加上overlap的大小，保存为.png格式
        initialfile=os.path.splitext(self.input_image_path)[0]+'_ovelap'+self.overlap
        self.output_image_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")],initialfile=initialfile)
        if self.output_image_path:
            image=self.segmented_image
            image.save(self.output_image_path)

    def calculate_class_pixel_ratios(self, segmented_output):
        #计算每个类别的像素比例
        class_counts = {}
        total_pixels = segmented_output.size
        for pixel_value in segmented_output.flatten():
            if pixel_value not in class_counts:
                class_counts[pixel_value] = 1
            else:
                class_counts[pixel_value] += 1

        class_pixel_ratios = {}
        for class_id, count in class_counts.items():
            class_pixel_ratios[class_id] = count / total_pixels

        return class_pixel_ratios

    def display_class_pixel_ratios(self,class_pixel_ratios):
        # 在标签上显示类别像素比例
        try:
            text = f"{class_pixel_ratios[1]*100:.2f}%"
        except KeyError:
            text = "0%"

        #更新含云量
        self.class_ratio_panel3.config(text=text)

    def start_segmentation(self):
        if self.input_image_path:
            # 加载模型
            args = parse_args()
            # 初始化模型
            model = init_model(args.config, args.checkpoint, device=args.device)
            if args.device == 'cpu':
                model = revert_sync_batchnorm(model)

            # 使用模型进行分割
            start_time = time.time()
            segmented_output = inference_model(model, self.input_image_path)
            end_time = time.time()
            self.class_ratio_panel2.config(text=f"{(end_time-start_time):.2f}s")

            # 处理分割结果
            segmented_output_data = segmented_output.pred_sem_seg.data[0]
            segmented_output_array = np.array(segmented_output_data)

            rgb_image_array = np.zeros((segmented_output_array.shape[0], segmented_output_array.shape[1], 3), dtype=np.uint8)
            rgb_image_array[segmented_output_array == 1] = [0, 255, 255]
            rgb_image_array[segmented_output_array == 0] = [0, 0, 0]
            self.segmented_image = Image.fromarray(rgb_image_array)
            self.mix_image = self.segmented_image

            segmented_show=self.segmented_image.resize((350, 350))  # 调整图片大小以适应窗口
            segmented_image_tk = ImageTk.PhotoImage(segmented_show)
            self.output_image_panel.config(image=segmented_image_tk)
            self.output_image_panel.image = segmented_image_tk

            #处理滑块
            self.slider.config(state=NORMAL)
            self.slider.set(1.00)

            # 计算各个类别像素比例
            class_pixel_ratios = self.calculate_class_pixel_ratios(segmented_output_array)
            self.display_class_pixel_ratios(class_pixel_ratios)
        else:
            print("请先加载图片")

    def update_blend(self, value):#更新主界面的分割图片展示区域
        # 获取滑块的值
        self.overlap=value
        alpha = float(value)
        # 创建混合图像
        self.mix_image = Image.blend(self.input_image, self.segmented_image, alpha)
        blended_image=self.mix_image.resize((350,350))
        # 更新 Label 显示混合图像
        blended_photo = ImageTk.PhotoImage(blended_image)
        self.output_image_panel.config(image=blended_photo)
        self.output_image_panel.image = blended_photo

    def update_mix(self, value): #更新canvas画布的图片展示区域
        self.overlap2 = value
        alpha = float(value)
        # 创建混合图像
        self.mix_photo = Image.blend(self.input_image, self.canvas_segmented, alpha)
        # 更新 Label 显示混合图像
        self.mix_photo = ImageTk.PhotoImage(self.mix_photo)
        self.canvas.itemconfigure(self.image_item, image=self.mix_photo)


    def create_canvas_window(self, event):
        # 创建画布的函数
        # 获取图片大小
        width, height = self.mix_image.size[0],self.mix_image.size[1]

        # 子窗口创建
        self.canvas_window = tk.Toplevel(self.root)
        self.canvas_window.title("Canvas")
        self.canvas_window.geometry(f"{width+50}x{height+100}")

        # 创建画布
        self.canvas = tk.Canvas(self.canvas_window, width=width, height=height, cursor="pencil")
        self.canvas.place(x=25, y=50, width=width, height=height)

        # 创建画布上的图片
        self.canvas_segmented = self.segmented_image.copy()#将segmented_image转移到canvas_segmented来处理

        self.mix_photo = self.mix_image.copy()
        self.mix_photo = ImageTk.PhotoImage(self.mix_photo)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.mix_photo)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.mix_photo)

        #画笔设置
        self.pencil_color = "#00FFFF"

        # 按钮
        button_frame = tk.Frame(self.canvas_window)
        button_frame.place(x=0, y=0, relwidth=1, height=50)

        save_button = tk.Button(button_frame, text="保存画布", command=self.save_canvas)
        save_button.place(relx=0.7,y=10,relwidth=0.2,height=30)

        self.label = tk.Label(button_frame, text="画笔大小")
        self.label.place(relx=0.1,y=10,relwidth=0.1,height=30)
        self.pencil_size = tk.IntVar(value=5)
        self.spinbox = tk.Spinbox(button_frame,from_=1, to=10, increment=1,textvariable=self.pencil_size)
        self.spinbox.place(relx=0.2,y=10,relwidth=0.1,height=30)

        self.pencil_color_Bool = tk.BooleanVar(value=True)
        self.add_button = tk.Radiobutton(button_frame, text="添加", value=True,
                                         variable=self.pencil_color_Bool,command=self.rgb_to_hex)
        self.add_button.place(relx=0.4,y=10,relwidth=0.1,height=30)
        self.remove_button = tk.Radiobutton(button_frame, text="去除", value=False,
                                            variable=self.pencil_color_Bool,command=self.rgb_to_hex)
        self.remove_button.place(relx=0.5,y=10,relwidth=0.1,height=30)

        self.slider2 = tk.Scale(self.canvas_window, from_=0.00, to=1.00, orient=tk.HORIZONTAL, command=self.update_mix,
                               resolution=0.05)
        self.slider2.set(float(self.overlap))
        self.slider2.place(relx=0.1, y=50+height, relwidth=0.8, height=45)

        self.canvas.bind("<B1-Motion>", self.draw_segmentation)

    def rgb_to_hex(self):
        #调整画笔的颜色
        if self.pencil_color_Bool.get():
            self.pencil_color="#00FFFF"
        else:
            self.pencil_color="#000000"

    def draw_segmentation(self, event):
        if self.mix_image:
            # 在画布上绘制用户的绘制动作
            x, y = event.x, event.y

            draw = ImageDraw.Draw(self.canvas_segmented)
            draw.ellipse([x, y, x+self.pencil_size.get(), y+self.pencil_size.get()],
                         fill=self.pencil_color, outline=self.pencil_color)

            # self.canvas.create_oval(x, y, x+self.pencil_size.get(), y+self.pencil_size.get(),
            #                         fill=self.rgb_to_hex(),outline=self.rgb_to_hex())

            self.update_mix(self.overlap2)

    def save_canvas(self):
        # 保存绘制的内容为Image对象
        self.segmented_image = self.canvas_segmented.copy()
        self.update_blend(self.overlap)

        segmented_output_array= np.array(self.segmented_image)
        segmented_output_array[(segmented_output_array == [0, 255, 255]).all(axis=2)] = 1
        class_pixel_ratios = self.calculate_class_pixel_ratios(segmented_output_array)
        self.display_class_pixel_ratios(class_pixel_ratios)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
