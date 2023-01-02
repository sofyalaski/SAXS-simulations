import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.gridspec as gridspec


def MSE(sampled, predicted):
    if len(sampled) == 0:
        return np.inf
    else:
        return mean_squared_error(sampled, predicted)

def set_value(row, true_shape, colname_true, colname_pred):
    if row.true_shape == true_shape:
        return row[colname_true]
    else:
        return row[colname_pred]
def set_label(row, true_shape, label_true, label_false):
    if row.true_shape == true_shape:
        return label_true
    else:
        return label_false



def plot_outcomes_identified(df, data_name):
    print('Accuracy is {0:.2%}'.format((df.true_shape == df.pred_shape).sum()/len(df)))
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.3)
    shapes = ['sphere', 'hardsphere', 'cylinder']
    # shape classified correctly
    correct = [((df.true_shape == 0) & (df.pred_shape == 0)).sum(),((df.true_shape == 1) & (df.pred_shape == 1)).sum(),((df.true_shape == 2) & (df.pred_shape == 2)).sum()]
    wrong = [((df.true_shape == 0) & (df.pred_shape != 0)).sum(),((df.true_shape == 1) & (df.pred_shape != 1)).sum(),((df.true_shape == 2) & (df.pred_shape != 2)).sum()]
    correct = [correct[i]/(correct[i]+ wrong[i])*100 for i in range(len(correct))]
    wrong = [100-correct[i] for i in range(len(wrong))]
    print(correct)
    ax = axes[0,0]
    ax.bar(shapes, correct, width=0.35, label='correct', color = 'peachpuff')
    ax.bar(shapes, wrong, width=0.35,bottom=correct, label='misclassified', color = 'plum')
    ax.set_ylabel('correct predictions, %')
    ax.set_title('Shapes predictions')
    ax.text(-0.05, 5, 'correct', rotation = 90)
    ax.text(0.95, 5, 'correct', rotation = 90)
    ax.text(1.95, 5, 'correct', rotation = 90)
    
    ax.text(-0.05, 65, 'misclassified', rotation = 90)
    ax.text(0.95, 65, 'misclassified', rotation = 90)
    ax.text(1.95, 65, 'misclassified', rotation = 90)

    # radius of a correctly identified shapes
    #print('Total radius MSE is {err:.2f}'.format(err= mean_squared_error(df.radius, df.pred_radius)))
    print('MSE for radius per shape for correctly identified instances as follows: sphere: {sMSE:.4f}, hardsphere: {hsMSE:.4f} and cylinder: {cMSE:.4f}'.format\
        (sMSE = MSE(df[(df.true_shape == 0) & (df.pred_shape == 0)].radius, df[(df.true_shape == 0) & (df.pred_shape == 0)].pred_radius), 
         hsMSE = MSE(df[(df.true_shape == 1) & (df.pred_shape == 1)].radius, df[(df.true_shape == 1) & (df.pred_shape == 1)].pred_radius),
         cMSE = MSE(df[(df.true_shape == 2) & (df.pred_shape == 2)].radius, df[(df.true_shape == 2) & (df.pred_shape == 2)].pred_radius)))
      
    #print('Total radius MSE is {err:.2f}'.format(err= mean_squared_error(df.radius, df.pred_radius)))
    print('MSE for radius polydispersity per shape for correctly identified instances as follows: sphere: {sMSE:.4f}, hardsphere: {hsMSE:.4f} and cylinder: {cMSE:.4f}'.format\
        (sMSE = MSE(df[(df.true_shape == 0) & (df.pred_shape == 0)].radius_pd, df[(df.true_shape == 0) & (df.pred_shape == 0)].pred_radius_pd), 
         hsMSE = MSE(df[(df.true_shape == 1) & (df.pred_shape == 1)].radius_pd, df[(df.true_shape == 1) & (df.pred_shape == 1)].pred_radius_pd),
         cMSE = MSE(df[(df.true_shape == 2) & (df.pred_shape == 2)].radius_pd, df[(df.true_shape == 2) & (df.pred_shape == 2)].pred_radius_pd)))

    print('MSE for cylinder length for correctly identified instances: {cMSE:.4f}'.format\
        (cMSE = MSE(df[(df.true_shape == 2) & (df.pred_shape == 2)].length, df[(df.true_shape == 2) & (df.pred_shape == 2)].pred_length)))
    print('MSE for cylinder length polydispersity for correctly identified instances: {cMSE:.4f}'.format\
        (cMSE = MSE(df[(df.true_shape == 2) & (df.pred_shape == 2)].length_pd, df[(df.true_shape == 2) & (df.pred_shape == 2)].pred_length_pd)))
    print('MSE for cylinder length for correctly identified instances: {hMSE:.4f}'.format\
        (hMSE = MSE(df[(df.true_shape == 1) & (df.pred_shape == 1)].volfraction, df[(df.true_shape == 1) & (df.pred_shape == 1)].pred_volfraction)))
    #stacked df to create violinplots
    df_stacked = df[df.true_shape ==df.pred_shape].drop(columns = ['pred_shape']).set_index('true_shape').stack().reset_index().rename(columns = {'level_1':'feature', 0:'value'})
    df_stacked.loc[df_stacked.true_shape == 0, 'true_shape'] = "sphere"
    df_stacked.loc[df_stacked.true_shape == 1, 'true_shape'] = "hardsphere"
    df_stacked.loc[df_stacked.true_shape == 2, 'true_shape'] = "cylinder"

    ax = axes[0,1]
    data = df_stacked[(df_stacked.feature == 'radius')|(df_stacked.feature == 'pred_radius')]
    data.loc[data.feature == 'radius', 'feature'] = "sampled"
    data.loc[data.feature == 'pred_radius', 'feature'] = "predicted"
    sns.violinplot(data = data.sort_values(by = ['true_shape','feature'], ascending = False), x="true_shape", y="value", hue="feature", split = False, ax=ax, palette=['peachpuff', 'plum'])
    ax.set_title('Radius distribution for correct shapes')
    ax.set_ylabel("radius, nm")
    ax.set_xlabel("")
    #ax.set_ylim([-5,10])
    ax.legend()

    ax = axes[0,2]
    data = df_stacked[(df_stacked.feature == 'radius_pd')|(df_stacked.feature == 'pred_radius_pd')]
    data.loc[data.feature == 'radius_pd', 'feature'] = "sampled"
    data.loc[data.feature == 'pred_radius_pd', 'feature'] = "predicted"
    sns.violinplot(data = data.sort_values(by = ['true_shape','feature'], ascending = False), x="true_shape", y="value", hue="feature", split = False, ax=ax, palette=['peachpuff', 'plum'])
    ax.set_title('Radius polidispersity distribution\nfor correct shapes')
    ax.legend()
    ax.set_ylabel("radius pd")
    ax.set_xlabel("")
    #ax.set_ylim([-1,1])

    # length
    df_stacked = df.set_index(['true_shape', 'pred_shape']).stack().reset_index().rename(columns = {'level_2':'feature', 0:'value'}).assign(y=1)

    ax = axes[1,0]
    data = df_stacked[((df_stacked.feature == 'length')&(df_stacked.true_shape ==2))|((df_stacked.feature == 'pred_length')&(df_stacked.pred_shape == 2))]
    data.loc[data.feature == 'length', 'feature'] = "sampled"
    data.loc[data.feature == 'pred_length', 'feature'] = "predicted"
    sns.violinplot(data = data.sort_values(by = 'feature', ascending = False), x="y",y = "value", hue="feature",split = False, ax=ax, palette=['peachpuff', 'plum'] )
    ax.set_title('Length distribution of cylinder')
    ax.set_xlabel("cylinder")
    ax.set_ylabel("length, nm")
    ax.set_xticks([])
    ax.get_legend().remove()
    ax.text(-.4, 57, "sampled")
    ax.text(0.2, 57, "predicted")
    #ax.set_ylim([-10,50])

    ax = axes[1,1]
    data = df_stacked[((df_stacked.feature == 'length_pd')&(df_stacked.true_shape ==2))|((df_stacked.feature == 'pred_length_pd')&(df_stacked.pred_shape == 2))]
    data.loc[data.feature == 'length_pd', 'feature'] = "sampled"
    data.loc[data.feature == 'pred_length_pd', 'feature'] = "predicted"
    sns.violinplot(data = data.sort_values(by = 'feature', ascending = False),  x="y",y = "value", hue="feature", split = False, ax=ax, palette=['peachpuff', 'plum'])
    ax.set_title('Length polidispersity distribution of cylinder')
    ax.set_xlabel("cylinder")
    ax.set_ylabel("length pd")
    ax.set_xticks([])
    ax.get_legend().remove()
    ax.text(-.4, 0.25, "sampled")
    ax.text(0.2, 0.25, "predicted")
    #ax.set_ylim([-1,1])

    ax = axes[1,2]
    data = df_stacked[((df_stacked.feature == 'volfraction')&(df_stacked.true_shape ==1))|((df_stacked.feature == 'pred_volfraction')&(df_stacked.pred_shape == 1))]
    data.loc[data.feature == 'volfraction', 'feature'] = "sampled"
    data.loc[data.feature == 'pred_volfraction', 'feature'] = "predicted"
    sns.violinplot(data = data.sort_values(by = 'feature', ascending = False), x="y",y = "value", hue="feature", split = False, ax=ax, palette=['peachpuff', 'plum'])
    ax.set_title('Volume fraction distribution of hardsphere')
    ax.set_xlabel("hardsphere")
    ax.set_ylabel("volumefraction")
    ax.set_xticks([])
    ax.get_legend().remove()
    ax.text(-.4, 0.3, "sampled")
    ax.text(0.2, 0.3, "predicted")
    #ax.set_ylim([-0,1])

    plt.suptitle('{d} Data'.format(d = data_name))
    



def describe_false_shapes(false_spheres, false_hardspheres, false_cylinders):
    plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.5)
    
    false_spheres['value'] = false_spheres.apply(set_value, args = ( 0, 'radius', 'pred_radius'), axis=1)
    false_spheres['feature'] = false_spheres.apply(set_label,args = (0, "FN", "FP"), axis=1)
    false_spheres['category'] = "sphere"

    false_hardspheres['value'] = false_hardspheres.apply(set_value, args = ( 1, 'radius', 'pred_radius'), axis=1)
    false_hardspheres['feature'] = false_hardspheres.apply(set_label,args =  (1, "FN", "FP"), axis=1)
    false_hardspheres['category'] = "hardsphere"

    false_cylinders['value'] = false_cylinders.apply(set_value, args = ( 2, 'radius', 'pred_radius'), axis=1)
    false_cylinders['feature'] = false_cylinders.apply(set_label,args =  (2, "FN", "FP"), axis=1)
    false_cylinders['category'] = "cylinders"

    false_radii = pd.concat([false_spheres[['value','feature', 'category']],false_hardspheres[['value','feature', 'category']],false_cylinders[['value','feature', 'category']]])

    ax1 = plt.subplot(gs[0, 1:3])
    sns.violinplot(data = false_radii.sort_values(by = ['category','feature'], ascending = False),x = 'category', y="value", hue="feature", split = False, ax=ax1, palette=['peachpuff', 'plum'])
    ax1.set_title('Radius distribution for undentified shapes')
    ax1.set_ylabel("radius, nm")
    ax1.set_xlabel("")
    ax1.legend()

    
    false_spheres['value'] = false_spheres.apply(set_value, args = ( 0, 'radius_pd', 'pred_radius_pd'), axis=1)
    false_spheres['feature'] = false_spheres.apply(set_label,args =  (0, "FN", "FP"), axis=1)
    false_spheres['category'] = "sphere"

    false_hardspheres['value'] = false_hardspheres.apply(set_value, args = ( 1, 'radius_pd', 'pred_radius_pd'), axis=1)
    false_hardspheres['feature'] = false_hardspheres.apply(set_label,args =  (1, "FN", "FP"), axis=1)
    false_hardspheres['category'] = "hardsphere"

    false_cylinders['value'] = false_cylinders.apply(set_value, args = ( 2, 'radius_pd', 'pred_radius_pd'), axis=1)
    false_cylinders['feature'] = false_cylinders.apply(set_label,args =  (2, "FN", "FP"), axis=1)
    false_cylinders['category'] = "cylinders"

    false_radii_pd = pd.concat([false_spheres[['value','feature', 'category']],false_hardspheres[['value','feature', 'category']],false_cylinders[['value','feature', 'category']]])

    ax2 = plt.subplot(gs[0, 3:5])
    sns.violinplot(data = false_radii_pd.sort_values(by = ['category','feature'], ascending = False), x="category", y="value", hue="feature", split = False, ax=ax2, palette=['peachpuff', 'plum'])
    ax2.set_title('Radius polidispersity distribution\nfor correct shapes')
    ax2.legend()
    ax2.set_ylabel("radius pd")
    ax2.set_xlabel("")

    # length
    
    false_cylinders['value'] = false_cylinders.apply(set_value, args = ( 2, 'length', 'pred_length'), axis=1)
    false_cylinders['feature'] = false_cylinders.apply(set_label,args =  (2, "FN", "FP"), axis=1)
    false_cylinders['category'] = "cylinders"

    ax3 = plt.subplot(gs[1, 0:2])
    sns.violinplot(data = false_cylinders[['value', 'feature','category']].sort_values(by = 'feature', ascending = False), x="category",y = "value", hue="feature",split = False, ax=ax3, palette=['peachpuff', 'plum'] )
    ax3.set_title('Length distribution of cylinder')
    ax3.set_xlabel("cylinder")
    ax3.set_ylabel("length, nm")
    ax3.set_xticks([])
    ax3.get_legend().remove()
    ax3.text(-.3, 25, "FP")
    ax3.text(0.3, 25, "FN")
    #ax3.set_ylim([-.5,6])

    ax4 = plt.subplot(gs[1, 2:4])
    
    false_cylinders['value'] = false_cylinders.apply(set_value, args = ( 2, 'length_pd', 'pred_length_pd'), axis=1)
    false_cylinders['feature'] = false_cylinders.apply(set_label,args =  (2, "FN", "FP"), axis=1)
    false_cylinders['category'] = "cylinders"

    sns.violinplot(data = false_cylinders[['value', 'feature','category']].sort_values(by = 'feature', ascending = False),  x="category",y = "value", hue="feature", split = False, ax=ax4, palette=['peachpuff', 'plum'])
    ax4.set_title('Length polidispersity distribution of cylinder')
    ax4.set_xlabel("cylinder")
    ax4.set_ylabel("length pd")
    ax4.set_xticks([])
    ax4.get_legend().remove()
    ax4.text(-.3, 12.5, "FP")
    ax4.text(0.3, 12.5, "FN")
    #ax4.set_ylim([-0.05, 0.35])

    ax5 = plt.subplot(gs[1, 4:6])
    false_hardspheres['value'] = false_hardspheres.apply(set_value, args = ( 1, 'volfraction', 'pred_volfraction'), axis=1)
    false_hardspheres['feature'] = false_hardspheres.apply(set_label,args =  (1, "FN", "FP"), axis=1)
    false_hardspheres['category'] = "hardsphere"


    sns.violinplot(data = false_hardspheres[['value', 'feature','category']].sort_values(by = 'feature', ascending = False), x="category",y = "value", hue="feature", split = False, ax=ax5, palette=['peachpuff', 'plum'])
    ax5.set_title('Volume fraction distribution of hardsphere')
    ax5.set_xlabel("hardsphere")
    ax5.set_ylabel("volume fraction")
    ax5.set_xticks([])
    ax5.get_legend().remove()
    #ax5.text(-.4, 0.5, "FN")
    #ax5.text(0.2, 0.5, "FP")
    #ax5.set_ylim([-0.1, 0.7])

    plt.suptitle('Parameter distributions for unidentified shapes')
    
    #plt.savefig("/home/slaskina/resultsML.svg", transparent=True)



def describe_positive_shapes(df_test):
    plt.figure(figsize=(16, 8))
    
    gs = gridspec.GridSpec(2, 6)    
    gs.update(wspace=0.5)

    spheres = df_test[df_test.pred_shape==0].copy()
    spheres['value'] = spheres.apply(set_value, args = (0, 'radius','pred_radius'),axis=1)
    spheres['feature'] = spheres.apply(set_label, args = (0, "TP", "FP"), axis=1)
    spheres['category'] = "sphere"

    hardspheres = df_test[df_test.pred_shape==1].copy()
    hardspheres['value'] = hardspheres.apply(set_value, args = (1, 'radius','pred_radius'),axis=1)
    hardspheres['feature'] = hardspheres.apply(set_label, args = (1, "TP", "FP"), axis=1)
    hardspheres['category'] = "hardsphere"

    cylinders = df_test[df_test.pred_shape==2].copy()
    cylinders['value'] = cylinders.apply(set_value, args = (2, 'radius','pred_radius'),axis=1)
    cylinders['feature'] = cylinders.apply(set_label, args = (2, "TP", "FP"), axis=1)
    cylinders['category'] = "cylinder"

    radii = pd.concat([spheres[['value','feature', 'category']],hardspheres[['value','feature', 'category']],cylinders[['value','feature', 'category']]])
    
    ax1 = plt.subplot(gs[0, 1:3])
    sns.violinplot(data = radii.sort_values(by = ['category' ,'feature'], ascending = False),x = 'category', y="value", hue="feature", split = False, ax=ax1, palette=['peachpuff', 'plum'])
    ax1.set_title('Radius distribution for undentified shapes')
    ax1.set_ylabel("radius, nm")
    ax1.set_xlabel("")
    ax1.legend()


    spheres = df_test[df_test.pred_shape==0].copy()
    spheres['value'] = spheres.apply(set_value, args = (0, 'radius_pd','pred_radius_pd'),axis=1)
    spheres['feature'] = spheres.apply(set_label, args = (0, "TP", "FP"), axis=1)
    spheres['category'] = "sphere"

    hardspheres = df_test[df_test.pred_shape==1].copy()
    hardspheres['value'] = hardspheres.apply(set_value, args = (1, 'radius_pd','pred_radius_pd'),axis=1)
    hardspheres['feature'] = hardspheres.apply(set_label, args = (1, "TP", "FP"), axis=1)
    hardspheres['category'] = "hardsphere"

    cylinders = df_test[df_test.pred_shape==2].copy()
    cylinders['value'] = cylinders.apply(set_value, args = (2, 'radius_pd','pred_radius_pd'),axis=1)
    cylinders['feature'] = cylinders.apply(set_label, args = (2, "TP", "FP"), axis=1)
    cylinders['category'] = "cylinder"

    radii_pd = pd.concat([spheres[['value','feature', 'category']],hardspheres[['value','feature', 'category']],cylinders[['value','feature', 'category']]])
    ax2 = plt.subplot(gs[0, 3:5])
    sns.violinplot(data = radii_pd.sort_values(by = ['category', 'feature'], ascending = False), x="category", y="value", hue="feature", split = False, ax=ax2, palette=['peachpuff', 'plum'])
    ax2.set_title('Radius polidispersity distribution\nfor correct shapes')
    ax2.legend()
    ax2.set_ylabel("radius pd")
    ax2.set_xlabel("")

    # length
    
    cylinders = df_test[df_test.pred_shape==2].copy()
    cylinders['value'] = cylinders.apply(set_value, args = (2, 'length','pred_length'),axis=1)
    cylinders['feature'] = cylinders.apply(set_label, args = (2, "TP", "FP"), axis=1)
    cylinders['category'] = "cylinder"

    ax3 = plt.subplot(gs[1, 0:2])
    sns.violinplot(data = cylinders[['value', 'feature','category']].sort_values(by = 'feature', ascending = False), x="category",y = "value", hue="feature",split = False, ax=ax3, palette=['peachpuff', 'plum'] )
    ax3.set_title('Length distribution of cylinder')
    ax3.set_xlabel("cylinder")
    ax3.set_ylabel("length, nm")
    ax3.set_xticks([])
    ax3.get_legend().remove()
    ax3.text(-.3, 50, "TP")
    ax3.text(0.3, 50, "FP")
    #ax3.set_ylim([-.5,6])



    cylinders = df_test[df_test.pred_shape==2].copy()
    cylinders['value'] = cylinders.apply(set_value, args = (2, 'length_pd','pred_length_pd'),axis=1)
    cylinders['feature'] = cylinders.apply(set_label, args = (2, "TP", "FP"), axis=1)
    cylinders['category'] = "cylinder"
    ax4 = plt.subplot(gs[1,2:4])

    sns.violinplot(data = cylinders[['value', 'feature','category']].sort_values(by = 'feature', ascending = False),  x="category",y = "value", hue="feature", split = False, ax=ax4, palette=['peachpuff', 'plum'])
    ax4.set_title('Length polidispersity distribution of cylinder')
    ax4.set_xlabel("cylinder")
    ax4.set_ylabel("length pd")
    ax4.set_xticks([])
    ax4.get_legend().remove()
    ax4.text(-.3, 11, "TP")
    ax4.text(0.3, 11, "FP")
    #ax4.set_ylim([-0.05, 0.35])


    hardspheres = df_test[df_test.pred_shape==1].copy()
    hardspheres['value'] = hardspheres.apply(set_value, args = (1, 'volfraction','pred_volfraction'),axis=1)
    hardspheres['feature'] = hardspheres.apply(set_label, args = (1, "TP", "FP"), axis=1)
    hardspheres['category'] = "hardsphere"

    ax5 = plt.subplot(gs[1,4:6])
    sns.violinplot(data = hardspheres[['value', 'feature','category']].sort_values(by = 'feature', ascending = False), x="category",y = "value", hue="feature", split = False, ax=ax5, palette=['peachpuff', 'plum'])
    ax5.set_title('Volume fraction distribution of hardsphere')
    ax5.set_xlabel("hardsphere")
    ax5.set_ylabel("volume fraction")
    ax5.set_xticks([])
    ax5.get_legend().remove()
    #ax5.text(-.4, 0.5, "FN")
    #ax5.text(0.2, 0.5, "FP")
    #ax5.set_ylim([-0.1, 0.7])

    plt.suptitle('Parameter distribution for shapes')
    
    #plt.savefig("/home/slaskina/resultsML.svg", transparent=True)