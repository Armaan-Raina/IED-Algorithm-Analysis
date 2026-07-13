from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree #Chosen due to similar nature with existing algorithm (a bunch of yes or no gates)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

FILEPATH = r"C:\\Users\\Armaa\\OneDrive\\Desktop\\IED-Algorithm-Analysis\\INSERTNAMEHERE.xlsx"


def process_excel(filepath: str) -> pd.DataFrame:
  '''
  Takes in excel workbook corresponding to all up to date ground truth labeled data
  and concatenates all but the first (summary) page in preparation for LSV and model training.
  '''
  all_sheets = pd.read_excel(filepath, sheet_name=None)
  
  data = pd.DataFrame()
  for k, v in all_sheets.items():
    #iterate through name, df
    if k == "Summary":
      continue
      #Not interested in the summary sheet
    
    #data actually starts at row 17, so we skip the first 16 rows
    entries = v.iloc[16:]

    data = pd.concat([data, entries], ignore_index=True)
  
  return data

def lsv(data: pd.DataFrame):
  fig, ax = plt.subplots(1,2,figsize=(14,4))
  X_reduced = PCA(n_components=2).fit_transform(data)
  zoom = 1.2

  views = [(30, 45), (15, 180), (45, 270)]
  for col, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(3, 3, col + 1, projection='3d')
    ax.set_facecolor('white')
    ax.set_box_aspect(None, zoom=zoom)
    scatter = ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        cmap=plt.get_cmap("coolwarm"),
        alpha=1,
        s=6,
        vmin=0, vmax=3,
        edgecolor='w',
        linewidth=0.25
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    fig.legend(
    labels=["IED", "Non IED"],
    fontsize=20,
    title_fontsize=0,
    frameon=False,
    loc='center right',
    bbox_to_anchor=(1.02, 0.5),
    handletextpad=0.5,
    labelspacing=0.8,
    )
    plt.savefig(r"C:\\Users\\Armaa\\OneDrive\\Desktop\\IED-Algorithm-Analysis\\figs\\lsv.png", dpi=300, bbox_inches='tight')


def importances(data: pd.DataFrame):
  sc = StandardScaler()
  feature_names = data.drop('Label', axis=1).columns
  X = sc.fit_transform(data.drop('Label', axis=1))
  y = data['Label']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  clf = tree.DecisionTreeClassifier()

  clf.fit(X_train, y_train)

  r_sens = permutation_importance(clf, X_test, y_test, scoring="recall", n_repeats=30) #Assess model sensitivity
  r_spec = permutation_importance(clf, X_test, y_test, scoring="precision", n_repeats=30) #Assess model specificity
  r_acc = permutation_importance(clf, X_test, y_test, n_repeats=30) #Assess model accuracy

  sens_order = r_sens.importances_mean.argsort()[::-1]
  spec_order = r_spec.importances_mean.argsort()[::-1]
  acc_order = r_acc.importances_mean.argsort()[::-1]

  sens_df = pd.DataFrame(
    r_sens.importances[sens_order].T,
    columns=feature_names[sens_order]
  )
  
  spec_df = pd.DataFrame(
    r_spec.importances[spec_order].T,
    columns=feature_names[spec_order]
  )

  acc_df = pd.DataFrame(
    r_acc.importances[acc_order].T,
    columns=feature_names[acc_order]
  )

  ax = sens_df.plot.box(vert=False, whis=10)

  ax.set_title("Permutation Importance (Sensitivity)")
  ax.axvline(x=0, color="k", linestyle="--")
  ax.set_xlabel("Decrease in Model Sensitivity")
  ax.set_ylabel("Feature")
  ax.figure.tight_layout()

  plt.savefig(r"C:\\Users\\Armaa\\OneDrive\\Desktop\\IED-Algorithm-Analysis\\figs\\sens_importances.png", dpi=300, bbox_inches='tight')

  ax = spec_df.plot.box(vert=False, whis=10)

  ax.set_title("Permutation Importance (Specificity)")
  ax.axvline(x=0, color="k", linestyle="--")
  ax.set_xlabel("Decrease in Model Specificity")
  ax.set_ylabel("Feature")
  ax.figure.tight_layout()

  plt.savefig(r"C:\\Users\\Armaa\\OneDrive\\Desktop\\IED-Algorithm-Analysis\\figs\\spec_importances.png", dpi=300, bbox_inches='tight')


  ax = acc_df.plot.box(vert=False, whis=10)

  ax.set_title("Permutation Importance (Accuracy)")
  ax.axvline(x=0, color="k", linestyle="--")
  ax.set_xlabel("Decrease in Model Accuracy")
  ax.set_ylabel("Feature")
  ax.figure.tight_layout()

  plt.savefig(r"C:\\Users\\Armaa\\OneDrive\\Desktop\\IED-Algorithm-Analysis\\figs\\acc_importances.png", dpi=300, bbox_inches='tight')
  
  

  
















