#include <igl/writeOFF.h>
#include "parameters.h"
#include "FractureSim.h"
#include "Gui.h"

class FractureGui : public Gui {
public:
	float m_dt = g_timestep;

	FractureSim *p_FractureSim = nullptr;

	FractureGui() {
		p_FractureSim = new FractureSim();
		setSimulation(p_FractureSim);

		// show vertex velocity instead of normal
		callback_clicked_vertex = [&](int clickedVertexIndex,
			int clickedObjectIndex,
			Eigen::Vector3d &pos,
			Eigen::Vector3d &dir) {
			RigidObject &o = p_FractureSim->getObjects()[clickedObjectIndex];
			pos = o.getVertexPosition(clickedVertexIndex);
			dir = o.getVelocity(pos);
		};
		start();
	}

	virtual void updateSimulationParameters() override {
		p_FractureSim->setTimestep(m_dt);
	}

	virtual void clearSimulation() override {

	}

	// TODO: define parameters which are used exactly by fracture simulation
	virtual void drawSimulationParameterMenu() override {
		ImGui::InputFloat("dt", &m_dt, 0, 0);
	}

	// TODO: legend on top-right corner, modify to show what we want to see
	virtual void drawSimulationStats() override {

	}
};

int main(int argc, char *argv[]) {
	new FractureGui();
	return 0;
}