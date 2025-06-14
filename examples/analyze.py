

    def analyze(self):
        """

        @param true_scale: true scale of aruco marker in centimeter!
        @return:
        """

        sf = np.zeros(len(self.P0) // 3)

        for i in range(2, len(self.P0) // 3 + 1):
            P0_i = self.P0.reshape(len(self.P0) // 3, 3)[:i]
            N_i = self.N.reshape(len(self.N) // 12, 4, 3)[:i]
            aruco_corners_3d = intersect_parallelized(P0=P0_i, N=N_i)
            aruco_dist = self.__evaluate(aruco_corners_3d)
            sf[i - 1] = (self.aruco_size) / aruco_dist

        plt.figure()
        plt.title("Scale Factor Estimation over Number of Images")
        plt.xlabel("# Images")
        plt.ylabel("Scale Factor")
        plt.grid()
        plt.plot(np.linspace(1, len(self.P0) // 3, len(self.P0) // 3)[1:], sf[1:])
        plt.show()