# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:20:44 2024

@author: jeconchao
"""


# %%
  folder = '09954'

   for folder in Path('data/output/').iterdir():

        s = pd.Series([re.match(r'\d+', i.name).group(0) for i in folder.iterdir()])
        s2 = pd.Series([re.search(r'p\d{1,2}', i.name).group(0)
                       for i in folder.iterdir()])
        s3 = pd.Series(
            [re.search(r'p\d{1,2}_\d{1,2}', i.name).group(0) for i in folder.iterdir()])
        df_check = pd.concat([s.rename('id_est'), s2.rename('preg'),
                              s3.rename('subpreg')], axis=1)

        n_est = df_check.id_est.nunique()
        subpregs = df_check.groupby('subpreg').id_est.count()

        df_check.groupby('id_est').preg.value_counts()

        nsubpreg_x_alumno = s.value_counts()

        if not nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)].empty:
            print(f'RBD {folder.name}:\n')
            print(nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)])
            print(subpregs[subpregs.ne(n_est)])
            print('\n')

    # %%

    e3 = Path('data/output')

    for n, i in enumerate(e3.rglob('*')):
        pass

    # %%

    cv2.imshow("Detected Lines", img_rptas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # %%

    # hsv_img = cv2.cvtColor(mask_naranjo,  cv2.COLOR_GRAY2BGR)
    # hsv_img = cv2.drawContours(mask_naranjo, big_contours, -1, (60, 200, 200), 3)
    cv2.imshow("Detected Lines", cv2.resize(im2, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%

    big_contours
    img_pregunta = bound_and_crop(media_img, c)

# %%

    img_crop = proc.recorte_imagen(cropped_img)
    img_crop_col = proc.procesamiento_color(img_crop)

    puntoy = proc.obtener_puntos(img_crop_col)

    for i in range(len(puntoy)-1):
        print(i)
        cropped_img_sub = img_crop[puntoy[i]:puntoy[i+1],]

        cv2.imshow("Detected Lines", cropped_img_sub)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def apply_approx(cnt):

    #     epsilon = 0.45*cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,epsilon,True)
    #     return approx

    # %%

    cv2.imshow("Detected Lines", cv2.resize(cropped_img, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
