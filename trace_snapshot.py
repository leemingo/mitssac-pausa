import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axes
from typing import Tuple
import matplotsoccer as mps
from skimage.transform import resize

class TraceSnapshot:
    def __init__(self, traces: pd.DataFrame, play_left_to_right: bool = True, field_dims: Tuple = (105, 68)):
        self.traces = traces.copy()
        self.field_dims = field_dims

        if play_left_to_right:
            x_cols = [c for c in self.traces.columns if c.endswith("_x")] # Away team
            y_cols = [c for c in self.traces.columns if c.endswith("_y")] # Away team
            self.traces[x_cols] = field_dims[0] - self.traces[x_cols].values
            self.traces[y_cols] = field_dims[1] - self.traces[y_cols].values

    @staticmethod
    def plot_players(xy: pd.DataFrame, ax: axes.Axes, obso: np.ndarray = None, field_dims: Tuple = (105, 68)):
        x = xy[xy.columns[0::2]].values[-1]
        y = xy[xy.columns[1::2]].values[-1]
        
        players = [c.split("_")[:2] for c in xy.columns[0::2]]
        color = "tab:red" if players[0][0] == "H" else "tab:blue" # Home team is red, Away team is blue

        ax.scatter(x, y, s=1500, c=color, edgecolors=color, linewidths=3, zorder=2) # Paper figure size: s=3000

        for p in players:
            player_xy = xy[[f"{p[0]}_{p[1]}_x", f"{p[0]}_{p[1]}_y"]].values            
            player_num = int(p[1])
            ax.plot(player_xy[:, 0], player_xy[:, 1], c=color, ls="--", zorder=0, lw=2.5) # Paper figure size: lw=5
  
            last_pos = player_xy[-1]
            annotation_text = str(player_num)

            if obso is not None:
                # 가장 obso가 높은 위치 하이라이팅
                obso_max_idx = np.unravel_index(np.argmax(obso, axis=None), obso.shape)
                obso_max_y, obso_max_x = obso_max_idx
                # obso 배열 크기를 필드 크기에 맞게 조정
                scale_x = field_dims[0] / obso.shape[1]
                scale_y = field_dims[1] / obso.shape[0]
                obso_max_x_field = obso_max_x * scale_x
                obso_max_y_field = obso_max_y * scale_y 
                
                # ax.scatter(
                #     obso_max_x_field,
                #     obso_max_y_field,
                #     s=1500,
                #     c="yellow",
                #     edgecolors="k",
                #     linewidths=2,
                #     marker="*",  # 별 모양
                #     zorder=2
                # )

            ax.annotate(
                annotation_text,    
                xy=last_pos,
                ha="center",
                va="center",
                color="w",
                fontsize=27, # Paper figure size: fontsize=40
                fontweight="bold",
                zorder=3,
            )


    def plot(self, ax= None, color ="green", save_format=None, obso: np.ndarray = None, vmin: float = 0, vmax: float = 1):
        if ax is None:
            figsize = (20.8, 14.4)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        mps.field(color, fig, ax, show=False)

        interpolated_obso = None # plot_players에 전달할 최종 (105, 68) 배열
        
        if obso is not None:
            target_shape_yx = (self.field_dims[1], self.field_dims[0]) # (68, 105)
            obso_resized_yx = resize(obso, target_shape_yx, 
                order=1, 
                mode='reflect', 
                anti_aliasing=True
            )

            interpolated_obso = obso_resized_yx#.T
            #print(f"vmin: {np.min(interpolated_obso):.3f}, vmax: {np.max(interpolated_obso):.3f}")
            im = ax.imshow(
                obso_resized_yx,
                origin='lower',
                zorder=-2,
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                cmap="Reds",
                alpha=1
            )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('value', fontsize=40)
            cbar.ax.tick_params(labelsize=40)

        traces = self.traces.dropna(axis=1, how="all")
        xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]
        team1_xy = traces[[c for c in xy_cols if c[0] == "H"]]
        team2_xy = traces[[c for c in xy_cols if c[0] == "A"]]
        
        if not team2_xy.empty:
            self.plot_players(team2_xy, ax, None, self.field_dims)
        if not team1_xy.empty:
            self.plot_players(team1_xy, ax, interpolated_obso, self.field_dims)
        
        if "B00_x" not in traces.keys():
            print("not exist ball trajectory")
            home_handle = plt.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=10, label='Home')
            away_handle = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=10, label='Away')
            ax.legend(handles=[home_handle, away_handle], loc='upper right', fontsize=30)
        else:
            ball_x = traces["B00_x"].values
            ball_y = traces["B00_y"].values
            # ax.scatter(ball_x[-1], ball_y[-1], s=600, c="black", edgecolors="k", marker="o", zorder=2)
            # ax.plot(ball_x[-30:], ball_y[-30:], "k", zorder=2)
            ax.scatter(ball_x[-1], ball_y[-1], s=1500, c="black", edgecolors="k", marker="o", zorder=2)
            ax.plot(ball_x[-50:], ball_y[-50:], "k", zorder=2, lw=5)
            home_handle = plt.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=40, label='Home')
            away_handle = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=40, label='Away')
            ball_handle = plt.Line2D([], [], color='k', marker='o', linestyle='None', markersize=40, label='Ball')
            if interpolated_obso is None:
                ax.legend(handles=[home_handle, away_handle, ball_handle], loc='upper right', fontsize=30)
            else:
                max_obso_handle = plt.Line2D([], [], color='yellow', marker='*', linestyle='None', markersize=40, label=f'Max OBSO\n ({np.max(interpolated_obso):.3f})')
                ax.legend(handles=[home_handle, away_handle, ball_handle, max_obso_handle], loc='upper right', fontsize=30)
                ax.legend(handles=[home_handle, away_handle, ball_handle], loc='upper right', fontsize=30)

        ax.set_title(f"PAUSA: {0.5247}", fontsize=40)
        if save_format == "pdf":
            plt.savefig(f'snapshot.pdf', bbox_inches="tight", dpi=300)
            plt.show()
        elif save_format == "png":
            plt.savefig(f'freeze_frame.png', bbox_inches="tight", dpi=300)
            plt.show()

        return fig, ax
    